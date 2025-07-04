# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import cv2

from DWT_IDWT_layer import DWT_1D, DWT_2D, IDWT_1D, IDWT_2D
from DWT_IDWT_Functions import DWTFunction_2D, IDWTFunction_2D
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling_train as sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets_wavelet as datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from dataset_load import LoadDataset

FLAGS = flags.FLAGS


def dwt_data(data):
    ###论文给了两种代码实现，这里用的第一种
    use_pytorch_wavelet = 0
    if not use_pytorch_wavelet:
        # 2位小波变换 一种比较简单的小波 haar(哈尔小波）
        dwt = DWT_2D("haar")
        iwt = IDWT_2D("haar")
    else:
        dwt = DWTFunction_2D.forward(J=1, mode='zero', wave='haar').cuda()  # 已改
        iwt = IDWTFunction_2D.backward(mode='zero', wave='haar').cuda()
    ##小波变换
    xll, xlh, xhl, xhh = dwt(data)
    ####论文输出小波变换为以下值
    dwt_data = torch.cat([xll, xlh, xhl, xhh], dim=1)  # [1, 4, 128, 128]
    dwt_data = np.squeeze(dwt_data)
    #####以下内容为保存小波变换的结果为mat文件
    # filepath = "./temp_data/"
    ####tensor转numpy
    # xll = xll.cpu().detach().numpy()
    # xlh = xlh.cpu().detach().numpy()
    # xhl = xhl.cpu().detach().numpy()
    # xhh = xhh.cpu().detach().numpy()
    # dwt_data = dwt_data.cpu().detach().numpy()
    # savemat("xll.mat", {'xll':xll})
    # savemat("xlh.mat", {'xlh':xlh})
    # savemat("xhl.mat", {'xhl':xhl})
    # savemat("xhh.mat", {'xhh':xhh})
    # savemat("dwt_data.mat", {'dwt_data': dwt_data})
    # return [xll,xlh,xhl,xhh]
    return dwt_data[1:, ...]

def idwt_data(ll, lw, hl, hh):
    ###小波分量的四个输出 合成原图像
    use_pytorch_wavelet = 0
    if not use_pytorch_wavelet:
        # 2位小波变换 一种比较简单的小波 haar(哈尔小波）
        iwt = IDWT_2D("haar")
    else:
        iwt = IDWTFunction_2D.backward(mode='zero', wave='haar').cuda()
    #########逆小波变换
    if not use_pytorch_wavelet:
        iwt_data = iwt(ll, lw, hl, hh)
    else:
        iwt_data = iwt((ll, [torch.stack(
            (lw, hl, hh), dim=2)]))
    #iwt_data = iwt_data.cpu().detach().numpy()
    #save_data("./temp_data/", "iwt_data.at", 'iwt_data', iwt_data)
    return iwt_data

def train(config, workdir):
    """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)
    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environment
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    # train_ds, eval_ds, _ = datasets.get_dataset(config,
    #                                          uniform_dequantization=config.data.uniform_dequantization)

    # transforms.Resize(config.data.image_size),
    tran_transform = transforms.Compose([

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    dataset = LoadDataset('/home/qgl/桌面/数据集/lsun_data/train_img120', tran_transform)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4)
    # train_iter = iter(dataloader)  # pytype: disable=wrong-arg-types
    # eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5

    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    step_end = 0
    logging.info("Starting training loop at step %d." % (initial_step,))
    for step in range(initial_step, num_train_steps + 1):
        for i, X in enumerate(dataloader):
            X = torch.mean(X, dim=1, keepdim=True)
            #print(X.shape) # 1 1 256 256
            X = X.cpu().numpy()  # 将张量从 GPU 移动到 CPU
            X = np.float32(X)
            X = torch.from_numpy(X).cuda()
            X = dwt_data(X)#3 128 128
            X = X[None, ...]
            #print(X.shape)  # 4 128 128

            step_end = step_end + 1
            # print(step_end)
            # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
            # batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
            batch = X.to(config.device).float()
            # batch=(batch-1)/2
            # print(batch.shape)
            # batch = batch.permute(0, 3, 1, 2)
            # batch = scaler(batch)
            # Execute one training step
            loss = train_step_fn(state, batch)
            if step_end % config.training.log_freq == 0:
                logging.info("step: %d, training_loss: %.5e" % (step_end, loss.item()))
                writer.add_scalar("training_loss", loss, step_end)

            # Save a temporary checkpoint to resume training after pre-emption periodically
            if step_end != 0 and step_end % config.training.snapshot_freq_for_preemption == 0:
                save_checkpoint(checkpoint_meta_dir, state)
            '''
      # Report the loss on an evaluation dataset periodically
      if step % config.training.eval_freq == 0:
        eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step_fn(state, eval_batch)
        logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
        writer.add_scalar("eval_loss", eval_loss.item(), step)
       '''
            # Save a checkpoint periodically and generate samples if needed
            if step_end != 0 and step_end % config.training.snapshot_freq == 0 or step_end == num_train_steps:
                # Save the checkpoint.
                save_step = (step_end // config.training.snapshot_freq)
                save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples
                if config.training.snapshot_sampling:
                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample, n = sampling_fn(score_model)
                    
                    # sample = np.squeeze(sample)
                    # xll, xlh, xhl, xhh =sample[0, :, :], sample[1, :, :], sample[2, :, :], sample[3, :, :]
                    # xll = xll[None,None,...]#1,1,384,384
                    # xlh = xlh[None,None,...]
                    # xhl = xhl[None,None,...]
                    # xhh = xhh[None,None,...]
                    # sample= idwt_data(xll,xlh,xhl,xhh)
                    
                    ema.restore(score_model.parameters())
                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step_end))
                    tf.io.gfile.makedirs(this_sample_dir)
                    nrow = int(np.sqrt(sample.shape[0]))
                    image_grid = make_grid(sample, nrow, padding=2)
                    sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite('{}/sample.png'.format(this_sample_dir), sample[0, ..., 0])
                    

                    
                    # with tf.io.gfile.GFile(
                    #         os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                    #     np.save(fout, sample)

                    # with tf.io.gfile.GFile(
                    #         os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                    #     save_image(sample, fout)
            if step_end == 5000001:
                assert 0


def evaluate(config,
             workdir,
             eval_folder="eval"):
    """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    tf.io.gfile.makedirs(eval_dir)

    # Build data pipeline
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                evaluation=True)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting

        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean,
                                       continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)

    # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                        uniform_dequantization=True, evaluation=True)
    if config.eval.bpd_dataset.lower() == 'train':
        ds_bpd = train_ds_bpd
        bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == 'test':
        # Go over the dataset 5 times when computing likelihood on the test dataset
        ds_bpd = eval_ds_bpd
        bpd_num_repeats = 5
    else:
        raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

    # Build the likelihood computation function when likelihood is enabled
    if config.eval.enable_bpd:
        likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (config.eval.batch_size,
                          config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    # Use inceptionV3 for images with resolution higher than 256.
    inceptionv3 = config.data.image_size >= 256
    inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        while not tf.io.gfile.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            time.sleep(60)
            try:
                state = restore_checkpoint(ckpt_path, state, device=config.device)
            except:
                time.sleep(120)
                state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(score_model.parameters())
        # Compute the loss function on the full evaluation dataset if loss computation is enabled
        if config.eval.enable_loss:
            all_losses = []
            eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
            for i, batch in enumerate(eval_iter):
                eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
                eval_batch = eval_batch.permute(0, 3, 1, 2)
                eval_batch = scaler(eval_batch)
                eval_loss = eval_step(state, eval_batch)
                all_losses.append(eval_loss.item())
                if (i + 1) % 1000 == 0:
                    logging.info("Finished %dth step loss evaluation" % (i + 1))

            # Save loss values to disk or Google Cloud Storage
            all_losses = np.asarray(all_losses)
            with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
                fout.write(io_buffer.getvalue())

        # Compute log-likelihoods (bits/dim) if enabled
        if config.eval.enable_bpd:
            bpds = []
            for repeat in range(bpd_num_repeats):
                bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
                for batch_id in range(len(ds_bpd)):
                    batch = next(bpd_iter)
                    eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
                    eval_batch = eval_batch.permute(0, 3, 1, 2)
                    eval_batch = scaler(eval_batch)
                    bpd = likelihood_fn(score_model, eval_batch)[0]
                    bpd = bpd.detach().cpu().numpy().reshape(-1)
                    bpds.extend(bpd)
                    logging.info(
                        "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (
                        ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
                    bpd_round_id = batch_id + len(ds_bpd) * repeat
                    # Save bits/dim to disk or Google Cloud Storage
                    with tf.io.gfile.GFile(os.path.join(eval_dir,
                                                        f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                           "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, bpd)
                        fout.write(io_buffer.getvalue())

        # Generate samples and compute IS/FID/KID when enabled
        if config.eval.enable_sampling:
            num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
            for r in range(num_sampling_rounds):
                logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

                # Directory to save samples. Different for each host to avoid writing conflicts
                this_sample_dir = os.path.join(
                    eval_dir, f"ckpt_{ckpt}")
                tf.io.gfile.makedirs(this_sample_dir)
                samples, n = sampling_fn(score_model)
                samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
                samples = samples.reshape(
                    (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
                # Write samples to disk or Google Cloud Storage
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, samples=samples)
                    fout.write(io_buffer.getvalue())

                # Force garbage collection before calling TensorFlow code for Inception network
                gc.collect()
                latents = evaluation.run_inception_distributed(samples, inception_model,
                                                               inceptionv3=inceptionv3)
                # Force garbage collection again before returning to JAX code
                gc.collect()
                # Save latent represents of the Inception network to disk or Google Cloud Storage
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(
                        io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
                    fout.write(io_buffer.getvalue())

            # Compute inception scores, FIDs and KIDs.
            # Load all statistics that have been previously computed and saved for each host
            all_logits = []
            all_pools = []
            this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
            stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
            for stat_file in stats:
                with tf.io.gfile.GFile(stat_file, "rb") as fin:
                    stat = np.load(fin)
                    if not inceptionv3:
                        all_logits.append(stat["logits"])
                    all_pools.append(stat["pool_3"])

            if not inceptionv3:
                all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
            all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

            # Load pre-computed dataset statistics.
            data_stats = evaluation.load_dataset_stats(config)
            data_pools = data_stats["pool_3"]

            # Compute FID/KID/IS on all samples together.
            if not inceptionv3:
                inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
            else:
                inception_score = -1

            fid = tfgan.eval.frechet_classifier_distance_from_activations(
                data_pools, all_pools)
            # Hack to get tfgan KID work for eager execution.
            tf_data_pools = tf.convert_to_tensor(data_pools)
            tf_all_pools = tf.convert_to_tensor(all_pools)
            kid = tfgan.eval.kernel_classifier_distance_from_activations(
                tf_data_pools, tf_all_pools).numpy()
            del tf_data_pools, tf_all_pools

            logging.info(
                "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
                    ckpt, inception_score, fid, kid))

            with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                                   "wb") as f:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
                f.write(io_buffer.getvalue())
