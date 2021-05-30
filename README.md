##  Processing OCT, OCT-A images.
### Traditional Image Processing, DL, ML
___
**ENV** : tfgpu_2.2.2 version <br>
**contact** : lllee.dh@gmail.com <br><br>
**🔥️🔥️ Still working 🔥️🔥️** <br><br>
**description**:
- [O] 1. make preprocessing method - minmax , max(/.255)
- [O] 2. add displaying histogram option to displayData.py - opt:'hist'
- [🔥️] 3. Image Processing
    - [_] 3-1) K-mean's binarization
    - [O] 3-2) Otsu's
    - [O] 3-3) EAT : Enhaced Adaptive Frangi Filter
    - [_] 3-4) Fuzzy
    - [_] 3-5) Region Growing
    - [O] 3-6) Anisotropic Diffusion Filter
    - [_] 3-7) Wavelet
    - [🔥️] 3-8) Curvelet

- [🔥️] 4. Deep Learning
    - [O] 4-1) U-NET
        > from [IntelAI-UNET](https://github.com/IntelAI/unet)
    - [_] 4-2) AE
    - [_] 4-3) VAE 
    - [_] 4-4) GAN 
    - [_] 4-5) VICCE : there's no data. so need to make the fake image from the og image.
