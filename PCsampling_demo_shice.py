import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

import cv2

##################################################################
#import sampling as sampling
import sampling_3h_shice as sampling
from sampling_3h_shice import ReverseDiffusionPredictor,LangevinCorrector,AnnealedLangevinDynamics ,EulerMaruyamaPredictor,AncestralSamplingPredictor
#from sampling_4h6 import iwt_data
import aapm_sin_ncsnpp_3h as configs_3h  
import aapm_sin_ncsnpp_wavelet as configs_A
##################################################################
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
sys.path.append('..')
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
import torch
import numpy as np

from utils import restore_checkpoint
from scipy.io import savemat, loadmat
import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
from sde_lib import VESDE, VPSDE, subVPSDE
import os.path as osp
import operator_1 as operator
if len(sys.argv) > 1:
  start = int(sys.argv[1])
  end = int(sys.argv[2])


checkpoint_num = [[23,24]]#25 3h,A
# print(checkpoint_num)
# assert False
def get_predict(num):
  if num == 0:
    return None
  elif num == 1:
    return EulerMaruyamaPredictor
  elif num == 2:
    return ReverseDiffusionPredictor

def get_correct(num):
  if num == 0:
    return None
  elif num == 1:
    return LangevinCorrector
  elif num == 2:
    return AnnealedLangevinDynamics


#checkpoint_num = [23,24,25,32,35,44]
# checkpoint_num = [6,8,10,12,14,16]

predicts = [2]
corrects = [1]
for predict in predicts:
  for correct in corrects:
    # for check_num in checkpoint_num:
      sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
      if sde.lower() == 'vesde':

        ckpt_filename_A = '/home/qgl/sanshe/exp_wavelet_lsun84/checkpoints/checkpoint_46.pth'
        ckpt_filename_3h = '/home/qgl/sanshe/exp_3h84/checkpoints/checkpoint_50.pth'
        assert os.path.exists(ckpt_filename_A)
        #assert os.path.exists(ckpt_filename_3h)
        config_3h = configs_3h.get_config()
        config_A = configs_A.get_config()
          
        sde_3h = VESDE(sigma_min=config_3h.model.sigma_min, sigma_max=config_3h.model.sigma_max, N=config_3h.model.num_scales)
        sde_A = VESDE(sigma_min=config_A.model.sigma_min, sigma_max=config_A.model.sigma_max, N=config_A.model.num_scales)
        sampling_eps = 1e-5
          
      # wavelet model
      batch_size = 1 #@param {"type":"integer"}
      
      config_A.training.batch_size = batch_size
      config_A.eval.batch_size = batch_size
      config_3h.training.batch_size = batch_size
      config_3h.eval.batch_size = batch_size
      
      random_seed = 0 #@param {"type": "integer"}

      sigmas_A = mutils.get_sigmas(config_A)
      sigmas_3h = mutils.get_sigmas(config_3h)
      
      model_A = mutils.create_model(config_A)
      model_3h = mutils.create_model(config_3h)
      
      optimizer_A = get_optimizer(config_A, model_A.parameters())
      optimizer_3h = get_optimizer(config_3h, model_3h.parameters())
      
      ema_A = ExponentialMovingAverage(model_A.parameters(),
                                    decay=config_A.model.ema_rate)
      ema_3h = ExponentialMovingAverage(model_3h.parameters(),
                                    decay=config_3h.model.ema_rate)
      
      state_A = dict(step=0, optimizer=optimizer_A,
                  model=model_A, ema=ema_A)
      state_3h = dict(step=0, optimizer=optimizer_3h,
                  model=model_3h, ema=ema_3h)
      
      state_A = restore_checkpoint(ckpt_filename_A, state_A, config_A.device)
      state_3h = restore_checkpoint(ckpt_filename_3h, state_3h, config_3h.device)
      
      ema_A.copy_to(model_A.parameters())
      ema_3h.copy_to(model_3h.parameters())
      
      #@title PC sampling
      # img_size = config_hh.data.image_size
      # channels = config_hh.data.num_channels
      # shape = (batch_size, channels, img_size, img_size)
      # predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
      # corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
      predictor = get_predict(predict) #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
      corrector = get_correct(correct) #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}

      snr = 0.16  #0.16 #@param {"type": "number"}
      n_steps = 1#@param {"type": "integer"}
      probability_flow = False #@param {"type": "boolean"}
      psnr_result = []
      ssim_result = []
      for j in range(0, 1, 1):
        psnr_n = []
        ssim_n = []
        psnr_tc_n = []
        ssim_tc_n = []
        for k in range(26,27):

            print(k)
            rec_data = cv2.imread(f'/home/qgl/sanshe/rec_img/review/18.5_5/bt/bz{k}.png',cv2.IMREAD_GRAYSCALE)

            tc = cv2.imread(f'/home/qgl/sanshe/rec_img/review/18.5_5/bt/bz{k}.png', cv2.IMREAD_GRAYSCALE)


            # # 假设你的.mat文件中存储的图像变量名为'image'
            # mat_file_path = f'/home/qgl/sanshe/rec_img/shice/12_22/bt/{k}.mat'
            # mat_data = loadmat(mat_file_path)
            # rec_data = mat_data['Image_bfOut']  # 提取图像数据
            #
            # # 假设你的.mat文件中存储的图像变量名为'image'
            # mat_file_path = f'/home/qgl/sanshe/rec_img/shice/12_22/bz/{k}.mat'
            # mat_data = loadmat(mat_file_path)
            # tc = mat_data['cropped_BP']  # 提取图像数据


            tc = np.array(tc)
            rec_data = np.array(rec_data)


            size = 80
            x = int(128 - size / 2)
            y = int(128 + size / 2)

            img = rec_data[x:y, x:y]
            padded_img = np.zeros((256, 256), dtype=np.float32)
            start_x = (256 - img.shape[1]) // 2
            start_y = (256 - img.shape[0]) // 2
            padded_img[start_y:start_y + img.shape[0], start_x:start_x + img.shape[1]] = img
            rec_data = padded_img

            min_22 = rec_data.min()
            max_22 = rec_data.max()
            rec_data = (rec_data - min_22) / (max_22 - min_22)

            min_tc = tc.min()
            max_tc = tc.max()
            tc = (tc - min_tc) / (max_tc - min_tc)

            rec_data=rec_data[None,None,...]
            tc = tc[None,None,...]

            #rec_data = rec_data[1079:1335,1279:1535]


            for i in range(1):
              #print('##################' + str(i) + '#######################')
              img_size = config_A.data.image_size
              channels = config_A.data.num_channels
              shape = (batch_size, channels, img_size, img_size)
              #print(shape)
              # sampling_fn = sampling.get_pc_sampler(sde_A, shape,predictor, corrector,None, snr,
              #                                   probability_flow=False,
              #                                   continuous=config_A.training.continuous,denoise=True,
              #                                   eps=sampling_eps,  device=config_A.device)
              sampling_fn = sampling.get_pc_sampler(sde_3h,sde_A, shape,predictor, corrector,None, snr, n_steps=1,
                                                probability_flow=False,
                                                continuous_3h=config_3h.training.continuous,
                                                continuous_A=config_A.training.continuous,denoise=True,

                                                eps=sampling_eps, device_3h=config_3h.device, device_A=config_A.device)

              sampling_fn(model_A,model_3h,data_ob=rec_data, tc=tc,k=k)#model_3h





