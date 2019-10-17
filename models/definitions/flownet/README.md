# FlowNet Models
We have attempted to convert the [NVIDIA FlowNet2](https://github.com/NVIDIA/flownet2-pytorch)
 models and weights **from pytorch over to Gluon**.

### Model Definition
Currently we have FlowNet2-S (`flownetS.py`) and FlowNet2-C (`flownetC.py`) working.

### Pre-trained weights
The original `.pth` pre-trained models can be downloaded from their 
[GitHub](https://github.com/NVIDIA/flownet2-pytorch).

We have converted two models into `.params`, which you can download from 
[Google Drive](https://drive.google.com/open?id=1wA3lPxSPc4rKQoz6-8Pr077MulnHx0Sf).

`FlowNet2-S_checkpoint.pth` --> `FlowNet2-S_checkpoint.params`

`FlowNet2-C_checkpoint.pth` --> `FlowNet2-C_checkpoint.params`

We store the models in the `/VidDet/models/definitions/flownet/weights/` directory.

### The Minor Differences
None so far (_that we've found_)...
