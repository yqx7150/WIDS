# 将散射的包导了进来，注意有一些可能用不上
from random import betavariate
import sys
from op import fused_act
sys.path.append('..')
import functools
import operator_1 as operator
import matplotlib.pyplot as plt
import torch
import numpy as np
import abc
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim, \
    mean_squared_error as compare_mse
import odl
import glob
import pydicom
#from cv2 import imwrite, resize
#from func_test import WriteInfo
from scipy.io import loadmat, savemat
from time import sleep
from DWT_IDWT_layer import DWT_1D, DWT_2D, IDWT_1D, IDWT_2D
import cv2
import os
import os.path as osp
from DWT_IDWT_Functions import DWTFunction_2D, IDWTFunction_2D
from scipy.stats import pearsonr
from torchvision.utils import save_image
from tvdenoise import tvdenoise
import scipy.io as io
from PIL import Image
_CORRECTORS = {}
_PREDICTORS = {}


def write_Datax(filedir, dataname, model_num, radio, psnr, ssim, mse):
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    with open(osp.join(filedir, dataname), "a+") as f:  # a+
        f.writelines(str(model_num) + ': ' + str(round(radio, 2)) + ',  ' + str(round(psnr, 4)) + ',  ' + str(
            round(ssim, 4)) + ',  ' + str(round(mse, 4)))
        f.write('\n')


def write_Data_mx(filedir, dataname, index, model_num, radio, psnr, ssim, mse):
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    with open(osp.join(filedir, dataname), "w+") as f:  # a+
        f.writelines(str(model_num) + '  (' + str(index) + ' ) :  ' + str(round(radio, 2)) + ',  ' + '[' + str(
            round(psnr, 4)) + ',  ' + str(round(ssim, 4)) + ',  ' + str(round(mse, 4)) + ']')
        f.write('\n')


def write_Data_mid(filedir, dataname, model_num, radio, num, psnr, ssim, mse):
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    with open(osp.join(filedir, dataname), "a+") as f:  # a+
        f.writelines(str(model_num) + ': ' + 'radio=' + str(round(radio, 2)) + ',  ' + str(num) + ' :' + '[' + str(
            round(psnr, 4)) + ',  ' + str(round(ssim, 4)) + ',  ' + str(round(mse, 4)) + ']')
        f.write('\n')


def write_images(x, image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)


def save_data(save_filepath, dataname, data_key, save_data):
    if not os.path.exists(save_filepath):
        os.mkdir(save_filepath)
    savemat(osp.join(save_filepath, dataname), {data_key: save_data})
def set_predict(num):
    if num == 0:
        return 'None'
    elif num == 1:
        return 'EulerMaruyamaPredictor'
    elif num == 2:
        return 'ReverseDiffusionPredictor'


def set_correct(num):
    if num == 0:
        return 'None'
    elif num == 1:
        return 'LangevinCorrector'
    elif num == 2:
        return 'AnnealedLangevinDynamics'
def dwt_data(data):
    ###论文给了两种代码实现，这里用的第一种
    use_pytorch_wavelet = 0
    if not use_pytorch_wavelet:
        # 2位小波变换 一种比较简单的小波 haar(哈尔小波）
        dwt = DWT_2D("haar")
    else:
        dwt = DWTFunction_2D.forward(J=1, mode='zero', wave='haar').cuda()
    ##小波变换
    xll, xlh, xhl, xhh = dwt(data)
    ####论文输出小波变换为以下值
    dwt_data = torch.cat([xll, xlh, xhl, xhh], dim=1)  # [1, 4, 256, 256]
    dwt_data = np.squeeze(dwt_data)
    #####以下内容为保存小波变换的结果为mat文件
   
    return dwt_data


def iwt_data(ll, lw, hl, hh):
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
    iwt_data = iwt_data.cpu().detach().numpy()
    save_data("./temp_data/", "iwt_data.at", 'iwt_data', iwt_data)
    return iwt_data
    #iwt_data_new = iwt_data(xll, xlh, xhl, xhh)
def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

    sampler_name = config.sampling.method  # pc
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(sde=sde,
                                      shape=shape,
                                      inverse_scaler=inverse_scaler,
                                      denoise=config.sampling.noise_removal,
                                      eps=eps,
                                      device=config.device)
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     inverse_scaler=inverse_scaler,
                                     snr=config.sampling.snr,
                                     n_steps=config.sampling.n_steps_each,
                                     probability_flow=config.sampling.probability_flow,
                                     continuous=config.training.continuous,
                                     denoise=config.sampling.noise_removal,
                                     eps=eps,
                                     device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


# ===================================================================== ReverseDiffusionPredictor
@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


# =====================================================================

@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


# ================================================================================================== LangevinCorrector
@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


# ==================================================================================================

@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


# ========================================================================================================

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)

def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)


# ========================================================================================================
def cal_pccs(x, y):
    """
  warning: data format must be narray
  :param x: Variable 1
  :param y: The variable 2
  :return: pccs
  """
    x_mean, y_mean = np.mean(x), np.mean(y)
    return np.sum((x - x_mean) * (y - y_mean)) / np.sqrt(
        np.sum((x - x_mean) * (x - x_mean)) * np.sum((y - y_mean) * (y - y_mean)))


def get_pc_sampler(sde_3h, sde_A, shape,predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous_3h=True, continuous_A=True,
                   denoise=True, eps=1e-3, device_3h='cuda', device_A='cuda'):
    """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
    predictor_update_fn_3h = functools.partial(shared_predictor_update_fn,
                                               sde=sde_3h,
                                               predictor=predictor,
                                               probability_flow=probability_flow,
                                               continuous=continuous_3h)
    corrector_update_fn_3h = functools.partial(shared_corrector_update_fn,
                                               sde=sde_3h,
                                               corrector=corrector,
                                               continuous=continuous_3h,
                                               snr=snr,
                                               n_steps=n_steps)
    predictor_update_fn_A = functools.partial(shared_predictor_update_fn,
                                              sde=sde_A,
                                              predictor=predictor,
                                              probability_flow=probability_flow,
                                              continuous=continuous_A)
    corrector_update_fn_A = functools.partial(shared_corrector_update_fn,
                                              sde=sde_A,
                                              corrector=corrector,
                                              continuous=continuous_A,
                                              snr=snr,
                                              n_steps=n_steps)

    def pc_sampler(model_A,model_3h,data_ob, tc,k):

                    with torch.no_grad():
                        x_A = sde_A.prior_sampling(shape).to(device_A)# 1,4,128,128
                        timesteps_A = torch.linspace(sde_A.T, eps, sde_A.N, device=device_A)
                        timesteps_3h = torch.linspace(sde_3h.T, eps, sde_3h.N, device=device_3h)

                        images_list = []

                        x_A = x_A.cpu()
                        x_A = np.float32(x_A)
                        x_A = torch.from_numpy(x_A).cuda()



                        data_ob = np.float32(data_ob)  # 1，1，256，256
                        data_ob = torch.from_numpy(data_ob).cuda()

                        dc = dwt_data(data_ob)  # 3h保真项，4，128，128
                        dcll, dclh, dchl, dchh = dc[0, :, :], dc[1, :, :], dc[2, :, :], dc[3, :, :]

                        dcll = dcll[None, None, ...]  # 1,1,128,128
                        dclh = dclh[None, None, ...]
                        dchl = dchl[None, None, ...]
                        dchh = dchh[None, None, ...]

                        start = 1000
                        end = 2000
                        size = 256
                        x = int(128 - size / 2)
                        y = int(128 + size / 2)

                        for i in range(start,end):#按散射来，可调整

                            t_A = timesteps_A[i]
                            vec_t_A = torch.ones(x_A.shape[0], device=t_A.device) * t_A

                            ###############predictor################
                            x0_A, dwt_x_A = predictor_update_fn_A(x_A, vec_t_A, model=model_A)  # 1,4,128,128
                            dwt_x_A = np.squeeze(dwt_x_A)#4,128,128

                            # # ########################提取每个通道######################
                            xll, xlh, xhl, xhh = dwt_x_A[0, :, :], dwt_x_A[1, :, :], dwt_x_A[2, :, :], dwt_x_A[3, :, :]
                            xll = xll[None, None, ...]  # 1,1,128,128
                            xlh = xlh[None, None, ...]
                            xhl = xhl[None, None, ...]
                            xhh = xhh[None, None, ...]


                            # # ########################DC######################

                            a = 0.1  # 或许可以调整
                            b = 0.1
                            c = 0.1
                            d = 0.1
                            xll = xll + a * dcll
                            xlh = xlh + b * dclh
                            xhl = xhl + c * dchl
                            xhh = xhh + d * dchh

                            # ########################corrector######################
                            dwt_x_A = torch.stack((xll, xlh, xhl, xhh), dim=1)
                            dwt_x_A = np.squeeze(dwt_x_A)
                            dwt_x_A = dwt_x_A[None, ...]


                            x1, dwt_x_A = corrector_update_fn_A(x=dwt_x_A, t=vec_t_A, model=model_A)  # 1,4,128,128
                            dwt_x_A = np.squeeze(dwt_x_A)  # 4,128,128
                            

                            # ########################提取每个通道######################
                            xll, xlh, xhl, xhh = dwt_x_A[0, :, :], dwt_x_A[1, :, :], dwt_x_A[2, :, :], dwt_x_A[3, :, :]
                            xll = xll[None, None, ...]  # 1,1,128,128
                            xlh = xlh[None, None, ...]
                            xhl = xhl[None, None, ...]
                            xhh = xhh[None, None, ...]

                            a = 0.1 # 或许可以调整
                            b = 0.1
                            c = 0.1
                            d = 0.1

                            # ########################DC######################
                            xll = xll + a * dcll
                            xlh = xlh + b * dclh
                            xhl = xhl + c * dchl
                            xhh = xhh + d * dchh

                            idwt_x = iwt_data(xll, xlh, xhl, xhh)
                            idwt_x = (idwt_x - idwt_x.min()) / (idwt_x.max() - idwt_x.min())
                            idwt_x = torch.from_numpy(idwt_x).cuda()

                            b= 0.01
                            c= 0.01
                            d= 0.01

                            ######################提取3h########################
                            x_A = dwt_data(idwt_x)
                            xll = x_A[:1, ...]
                            xll = xll[None,...]

                            x_3h_ori = x_A[1:, ...]

                            x_3h = x_3h_ori[None, ...] # 1 3 128 128

                            t_3h = timesteps_3h[i]
                            vec_t_3h = torch.ones(x_3h.shape[0], device=t_3h.device) * t_3h

                            ###############predictor################
                            x01, x_3h = predictor_update_fn_3h(x_3h, vec_t_3h, model=model_3h)
                            x_3h = np.squeeze(x_3h)
                            xlh, xhl, xhh = x_3h[0, :, :], x_3h[1, :, :], x_3h[2, :, :]
                            xlh = xlh[None, None, ...]
                            xhl = xhl[None, None, ...]
                            xhh = xhh[None, None, ...]

                            xlh = xlh + b * dclh
                            xhl = xhl + c * dchl
                            xhh = xhh + d * dchh

                            # ########################corrector######################
                            x_3h = torch.stack((xlh, xhl, xhh), dim=1)
                            x_3h = np.squeeze(x_3h)
                            x_3h = x_3h[None, ...]


                            x01, x_3h = corrector_update_fn_3h(x_3h, vec_t_3h, model=model_3h)

                            x_3h = np.squeeze(x_3h)  # 3,128，128

                            xlh, xhl, xhh = x_3h[0, :, :], x_3h[1, :, :], x_3h[2, :, :]
                            xlh = xlh[None, None, ...]
                            xhl = xhl[None, None, ...]
                            xhh = xhh[None, None, ...]

                            xlh = xlh + b * dclh
                            xhl = xhl + c * dchl
                            xhh = xhh + d * dchh


                            xlh_ori, xhl_ori, xhh_ori = x_3h_ori[0, :, :], x_3h_ori[1, :, :], x_3h_ori[2, :, :]
                            xlh_ori = xlh_ori[None, None, ...]
                            xhl_ori = xhl_ori[None, None, ...]
                            xhh_ori= xhh_ori[None, None, ...]

                            t = 0.5
                            xlh = t * xlh_ori + (1-t)*xlh
                            xhl = t * xhl_ori +(1-t)* xhl
                            xhh = t * xhh_ori + (1-t)*xhh

                            idwt_x = iwt_data(xll, xlh, xhl, xhh)  # 1，1,256,256
                            idwt_x = torch.from_numpy(idwt_x).cuda()

                            idwt_x = (idwt_x - torch.min(idwt_x)) / (torch.max(idwt_x) - torch.min(idwt_x))

                            idwt_x = tvdenoise(idwt_x, 10, 2)


                            x_A = dwt_data(idwt_x)
                            x_A = x_A[None, ...]#1，4，128，128

                            idwt_x = np.squeeze(idwt_x)
                            idwt_x = idwt_x.cpu().numpy()

                            images_list.append(idwt_x[x:y,x:y].copy())

                            save_path = './review_ours_shice/'
                            os.makedirs(save_path, exist_ok=True)
                            filename = f'img_{i}.png'
                            save_path = os.path.join(save_path, filename)
                            Image.fromarray((idwt_x[x:y, x:y] * 255).astype(np.uint8)).save(save_path)

                        # save_path = './review_ours/review/18.5_5/'
                        # os.makedirs(save_path, exist_ok=True)
                        # filename = f'img_{k}.png'
                        # save_path = os.path.join(save_path, filename)
                        # Image.fromarray((idwt_x[x:y,x:y] * 255).astype(np.uint8)).save(save_path)


                            # save_path = './sampling_img_shice1_44'
                            # os.makedirs(save_path, exist_ok=True)
                            # filename = f'{i}.mat'  # 修改文件扩展名为.mat
                            # save_path = os.path.join(save_path, filename)

                            # 将图像数据乘以255并转换为uint8类型，然后保存为.mat文件
                            # 这里假设data已经是0-1之间的浮点数
                            # data_uint8 = (idwt_x * 255).astype(np.uint8)
                            # savemat(save_path, {'image': data_uint8})  # 将图像数据以'image'作为键保存在.mat文件中


                        print("ok")
                        return
    return pc_sampler


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
    """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x)

            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
