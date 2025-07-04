import numpy as np
import torch.fft as fft
import torch
def forward(obj,H):
       # obj=obj.cpu().numpy()
        FO=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(obj)))
        #FO=FO[:,:,None]
        #FO=np.array(FO,dtype=np.complex64)
        #FO=FO[:,:,None]
        I=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(FO*H)))
        I=np.real(I)
        I=np.array(I,dtype=np.float32)
        I=torch.as_tensor(I).cuda()
        return I
def backward(rec_data, mask_data):
    #rec_data = rec_data.cpu().numpy()
    rec_data_fft = np.fft.fft2(rec_data)
    mask_data_fft = np.fft.fft2(mask_data)
    k = 300  # 3000#1000
    C = (np.mean(mask_data[:]) * k) ** 2
    aa = rec_data_fft * np.conj(mask_data_fft)  # 512*512*3
    # aa = np.multiply(rec_data_fft,mask_data_fft)
    bb = (abs(mask_data_fft ** 2) + C).astype(dtype=np.float64)
    rec_img_fft = np.divide(aa, bb)
    rec_img = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(rec_img_fft))))
    rec_img = rec_img / np.max(rec_img)
   #rec_img = rec_img[896:1152, 896:1152]
    rec_img = np.array(rec_img, dtype=np.float32)
    rec_img = torch.as_tensor(rec_img).cuda()
    # rec_img = np.array(rec_img, dtype=np.float32)
    # rec_img = torch.as_tensor(rec_img).cuda()
    return rec_img
def forward_torch(obj,H):
        FO=fft.fftn(obj)
        I=fft.ifftn(FO*H)
        I=torch.real(I)
        return I
def backward_torch(I,H):
	FI=fft.fftn(I)#0.01
	OR=fft.ifftn(FI/H)
	OR=torch.real(OR)
	return OR
