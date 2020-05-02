# FERAtt: Facial Expression Recognition with Attention Net
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository is under construction ...

<div>
<div align="center" >
<img src="rec/emotion.gif" width="300">
</div>
</div>


### [Paper](http://openaccess.thecvf.com/content_CVPRW_2019/html/MBCCV/Fernandez_FERAtt_Facial_Expression_Recognition_With_Attention_Net_CVPRW_2019_paper.html) | [arXiv](https://arxiv.org/abs/1810.12121)

[Pedro D. Marrero Fernandez](https://pedrodiamel.github.io/)<sup>1</sup>, [Fidel A. Guerrero-Peña](https://www.linkedin.com/in/fidel-alejandro-guerrero-pe%C3%B1a-602634109/)<sup>1</sup>, [Tsang Ing Ren](https://www.linkedin.com/in/ing-ren-tsang-6551371/)<sup>1</sup>, [Alexandre Cunha](http://www.cambia.caltech.edu/cambia.html)<sup>2</sup>

- 1 Centro de Informatica (CIn), Universidade Federal de Pernambuco (UFPE), Brazil
- 2 Center for Advanced Methods in Biological Image Analysis (CAMBIA) California Institute of Technology, USA

Introduction
------------

Pytorch implementation for FERAtt neural net. Facial Expression Recognition with Attention Net (FERAtt), is based on the dual-branch architecture and consists of four major modules: (i) an attention module $$G_{att}$$ to extract the attention feature map, (ii) a feature extraction module $G_{ft}$ to obtain essential features from the input image $I$, (iii) a reconstruction module $G_{rec}$ to estimate a good attention image $I_{att}$, and (iv) a representation module $G_{rep}$ that is responsible for the representation and classification of the facial expression image.


<div align="center">
<img src="rec/feratt_arq.png" width="1024">
</div>



## Prerequisites

- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA cuDNN
- PyTorch 1.5

Installation
------------

    $git clone https://github.com/pedrodiamel/pytorchvision.git
    $cd pytorchvision
    $python setup.py install
    $pip install -r installation.txt

### Visualize result with Visdom

We now support Visdom for real-time loss visualization during training!

To use Visdom in the browser:

    # First install Python server and client
    pip install visdom
    # Start the server (probably in a screen or tmux)
    python -m visdom.server -env_path runs/visdom/
    # http://localhost:8097/


How use
------------

### Step 1: Train

    ./train_bu3dfe.sh
    ./train_ck.sh



Citation
------------

If you find this useful for your research, please cite the following paper.

```
@InProceedings{Fernandez_2019_CVPR_Workshops,
author = {Marrero Fernandez, Pedro D. and Guerrero Pena, Fidel A. and Ing Ren, Tsang and Cunha, Alexandre},
title = {FERAtt: Facial Expression Recognition With Attention Net},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```


Acknowledgments
------------

Gratefully acknowledge financial support from the Brazilian government agency FACEPE.

