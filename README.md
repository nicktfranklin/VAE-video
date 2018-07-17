# Variational Autoencoder embedding of Video from Zacks et al., 2006

The project here is to reduce the dimensionality of the video stimuli used by 
Zacks, Swallow, Vettel, & McAvoy, (2006) 
[[Paper](https://pages.wustl.edu/files/pages/imce/dcl/zacksbrainres06visualmotion.pdf)]. This is done by training
a variational autoenconder on still frames of the video and then using the low dimensional latent space as our 
embeddings.

Specifically, an MMD-Variational autoencoder (Zhao, 2017; described in the following
[Blog Post](http://szhao.me/2017/06/10/a-tutorial-on-mmd-variational-autoencoders.html) and
[Paper](https://arxiv.org/abs/1706.02262)) is trained on the set of frames from the six videos in Zacks et al., 2006.

This implementation varies slightly from 
[this pytorch implementation](https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb)
 by [Shubanshu Mishra](https://github.com/napsternxg).
 The original tensorflow implmentation by Shengjia Zhao can be found 
 [here](https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.ipynb)
 
## Files

* `preprocess_video.py` converts the video files into a numpy array and downsamples them from the original dimensions
of 240x320x3 per frame to 64x64x3 per frame. Downsampling is done to lower computation cost.
* `pytorch_vae.py` runs the VAE on the preprocessed data and outputs `video_color_Z_embedded_64.npy`,
    the embedded videos as one long concatenated array and `video_color_X_reconstructed_64.npy`, 
    which is a numpy array of the reconstructed video
* `samples_batch_33000.png` is a sample from the generative distribution of a trained network. This is blurry when 
compared to the reconstructed video but is useful for assessing training.

Because the data files are relatively large (1GB+), none of the processed data are strored here.
     

## Requirements
The VAE is best run on a machine with an NVIDIA GPU and `pytorch` installed. In addition  `numpy` and `opencv` are 
used to convert the videos into a readable formate. `matplotlib` is used to generate figures from VAE samples and 
reconstructed images.


## Additional Files
The raw video datasets can be found here: [video data](https://wustl.app.box.com/s/s7y29vxxv0ryvw9evih7dhmzuc6du4d8)