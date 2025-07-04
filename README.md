# WIDS

**Paper**: Wavelet-refinement-inspired diffusion model for scattering imaging

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

# The scattering imaging system
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
Simulation Test
```bash
python PCsampling_demo.py
```
Experiment Test
```bash
python PCsampling_demo_shice.py
```
# Checkpoints
WIDS : We provide pretrained checkpoints. You can download pretrained models from [Baidu cloud] (https://pan.baidu.com/s/1CZLfDmLZeSTBFnwx2Hmwbg) Extract the code (1230)

# Other Related Projects
  * Imaging through scattering media via generative diffusion model  
[<font size=5>**[Paper]**</font>](https://pubs.aip.org/aip/apl/article/124/5/051101/3176612/Imaging-through-scattering-media-via-generative )   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/ISDM)

  * Diffusion Model-boosted Multiplane Extrapolation for Digital Holographic Reconstruction  
[<font size=5>**[Paper]**</font>](https://doi.org/10.1364/OE.531147)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DMEDH)    

  * Dual-domain Mean-reverting Diffusion Model-enhanced Temporal Compressive Coherent Diffraction Imaging  
[<font size=5>**[Paper]**</font>](https://doi.org/10.1364/OE.517567)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DMDTC)
  
  * High-resolution iterative reconstruction at extremely low sampling rate for Fourier single-pixel imaging via diffusion model  
[<font size=5>**[Paper]**</font>](https://doi.org/10.1364/OE.510692)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/FSPI-DM)

  * Real-time intelligent 3D holographic photography for real-world scenarios  
[<font size=5>**[Paper]**</font>](https://doi.org/10.1364/OE.529107)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Intelligent-3D-holography)

