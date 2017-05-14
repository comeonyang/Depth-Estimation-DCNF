# Depth from Single Monocular Images

- This is the prediction/test code for the paper:

`Learning Depth from Single Monocular Images Using Deep Convolutional Neural Fields`;
available at: [http://arxiv.org/abs/1502.07411](http://arxiv.org/abs/1502.07411).

- This code is tested on Ubuntu 14.04, and requires Matlab 2014a, CUDA 6.5 or later versions.
  Tested GPUs are NVIDIA Titan Black, K40c, GTX 780.

- [Download this repository](https://bitbucket.org/fayao/dcnf-fcsp/get/f66628a4a991.zip)

- If this code is useful for your research, please consider to cite our work:
```
 @inproceedings{Depth2015CVPR,
    author = {Fayao Liu and Chunhua Shen and Guosheng Lin},
    title  = {Deep Convolutional Neural Fields for Depth Estimation from a Single Image},
    booktitle= {Proc. IEEE Conf. Computer Vision and Pattern Recognition},
    year   = {2015},
    url    = {http://arxiv.org/abs/1411.6387},
    pages  = {},
}
```
```
 @article{Depth2015Liu,
    author = {Fayao Liu and Chunhua Shen and Guosheng Lin  and Ian Reid},
    title  = {Learning Depth from Single Monocular Images Using Deep Convolutional Neural Fields},
    journal= {IEEE T. Pattern Analysis and Machine Intelligence},
    volume = {},
    number = {},
    year   = {2015},
    url    = {http://arxiv.org/abs/1502.07411},
    month  = {},
    pages  = {},
}
```




# Install

Two toolboxes are required for using this code. For convenience, they are included in the folder:

`./libs` and pre-compiled in Linux. These toolboxes are as follows:

1. MatConvNet is required for the CNN training, which can be downloaded at: http://www.vlfeat.org/matconvnet/

2. VLFeat is required for generating superpixels, which is available at http://www.vlfeat.org/.
This code is tested using the VLFeat 0.9.18 version.



# Run

1. Users need to compile MatConvNet before running our code.
Please refer to: http://www.vlfeat.org/matconvnet/

2. We provide a demo file in folder `./demo/`:

     `demo_DCNF_FCSP_depths_prediction.m`

	 This is a demo for predicting depths of given images using our trained model.

3. We provide two trained models (trained using the Make3D and NYUD2 datasets respectively) in the folder `./model_trained`.




# Contact

Professor Chunhua Shen (University of Adelaide, Australia)

email: <chunhua.shen@adelaide.edu.au>



# Copyright

Copyright (c) The authors, 2015.

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


** This code provided here is for non-commercial research purposes only! **

For commercial applications, please contact Chunhua Shen <http://www.cs.adelaide.edu.au/~chhshen/>.

2015

