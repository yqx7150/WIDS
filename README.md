# WIDS

**Paper**: WIDS: Wavelet-refinement-inspired diffusion model for scattering imaging

**Authors**: Xinyi Wu, Meng Teng, Qi Yu, Xinmin Ding, Wenbo Wan*, and Qiegen Liu*   

Optics and Laser Technology     

Date : Jul-4-2025  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2025, School of Information Engineering, Nanchang University.  
# Abstract
Scattering media causes the random refraction of light along their propagation paths, which notably diminishes the clarity of optical imaging. Current techniques predominantly focus on simple targets, thereby limiting their practical applicability in complex scenarios. This work proposes an approach for wavelet-refinement-inspired diffusion model for scattering imaging. A fullfrequency component diffusion model is utilized to extract priori information of global distribution, while a high-frequency component diffusion model is utilized to acquire priori information about the details of the target. In the reconstruction process, the trained models provide multi-scale constraints in iterations of reconstruction, with the physics-based deconvolution providing fidelity. The results indicate that this work outperforms traditional methods in the reconstruction of complex targets while exhibits robust generalization capabilities. Simulation and experimental validation show that the proposed method can effectively remove the gridding artifacts in the reconstructed images for complex targets. The average PSNR and SSIM of the reconstructed image can reach 22.49 dB and 0.78, respectively. The highest resolution of the algorithm can reach 28.51 lp/mm.

# Main procedure and performance
![Flowchart of WIDS](/Figures/4.png "Main procedure and performance")

![Simulation](/Figures/2.png "Main procedure and performance")

![Simulation_crossdata](/Figures/1.png "Main procedure and performance")

![Experiment](/Figures/5.png "Main procedure and performance")

![Spatial resolution](/Figures/3.png "Main procedure and performance")

# Optical system configuration.
![Structural diagram](/Figures/7.png "Optical system configuration")
![Imaging system](/Figures/6.png "Optical system configuration")

# Training
Full-frequency Diffusion Model
```bash
python main_wavelet.py --config=aapm_sin_ncsnpp_wavelet.py --workdir=exp_wavelet --mode=train --eval_folder=result
```

High-frequency Diffusion Model 
```bash
python main_3h.py --config=aapm_sin_ncsnpp_3h.py --workdir=exp_3h --mode=train --eval_folder=result
```
# Test
```bash
python PCsampling_demo.py
```

