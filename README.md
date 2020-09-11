# Neural Style Transfer
Implementation of Neural Style Transfer in Pytorch :heart:

Here, I have tried to implement the Neural Style Transfer with some modifications from [Gatys et al.](https://arxiv.org/abs/1508.06576).  

Below is a result, where I used picture of Lauren Mayberry for Content Image and One of the Vincent Van Gogh's painting as the style image and the result turned out to be pretty good painting of Lauren Mayberry in Van Gogh painting style.  
![Neural Style Transfer](./ims/style_transfer.png)

### Files
* __helper.py__: Contains the helper functions for loading the image and converting it to pytorch tensor and for converting tensor back to image format.  
* __layers.py__: Contains the dictionary of the layers from which the features are to be extracted and the weights for style layers when generating the image.  
* __main.py__: The script which takes care of forward pass and optimisation of the image.  
* __utils.py__: Contains functions for extracting features from the chosen layers which are present in the dictionary of ```helper.py``` and a small function to calculate the gram matrix of the tensor.  

### Usage
The script required PyTorch (This code was written with PyTorch ver. 1.2.0) and a CUDA enabled GPU will be nice as running it on CPU is not feasible but still as an option in the script.  
To run the use ```main.py``` with the following arguments:  
> ```--content```: Path/Link to the content image  
> ```--style```: Path/Link to the style image  
> ```--result```: Path to where the result image is to be saved  
> ```--size``` (optional, default= 512px max): The desired resolution for the generated image  
> ```--steps``` (optional, default= 4000): Number of iterations on the image
> ```--alpha`` (optional, default= 1): Weight for the content loss  
> ```--beta`` (optional, default= 100): Weight for the style loss  
> ```--lr``` (optional, default= 0.003): The Learning Rate of the optimizer
> ```--device``` (optional, default= _cpu_): The device on which the model is supposed to carry out the style transfer  
  
For those who want to tinker with the layers which are chosen for the feature extraction or use some different weights for style layers head over to ```layers.py``` and make the changes.   
Happy Coding!!!