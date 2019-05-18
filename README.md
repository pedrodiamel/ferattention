# FERAtt: Facial Expression Recognition with Attention Net
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


<div>
<div align="center" >
<img src="rec/emotion.gif" width="640">
</div>
</div>


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
- PyTorch 0.4

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
@article{fernandez2019feratt,
  title={FERAtt: Facial Expression Recognition with Attention Net},
  author={Fernandez, Pedro D Marrero and Pe{\~n}a, Fidel A Guerrero and Ren, Tsang Ing and Cunha, Alexandre},
  journal={arXiv preprint arXiv:1902.03284},
  year={2019}
}
```


Acknowledgments
------------

Gratefully acknowledge financial support from the Brazilian government agency FACEPE.

