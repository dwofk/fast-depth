FastDepth
============================

This repo offers trained models and evaluation code for the [FastDepth](http://fastdepth.mit.edu/) project at MIT.

<p align="center">
	<img src="img/visualization.png" alt="photo not available" width="50%" height="50%">
</p>

## Contents
0. [Requirements](#requirements)
0. [Trained Models](#trained-models)
0. [Evaluation](#evaluation)
0. [Deployment](#deployment)
0. [Results](#results)
0. [Citation](#citation)

## Requirements
- Install [PyTorch](https://pytorch.org/) on a machine with a CUDA GPU. Our code was developed on a system running PyTorch v0.4.1.
- Install the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) format libraries. Files in our pre-processed datasets are in HDF5 format.
  ```bash
  sudo apt-get update
  sudo apt-get install -y libhdf5-serial-dev hdf5-tools
  pip3 install h5py matplotlib imageio scikit-image opencv-python
  ```
- Download the preprocessed [NYU Depth V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset in HDF5 format and place it under a `data` folder outside the repo directory. The NYU dataset requires 32G of storage space.
	```bash
	mkdir data; cd data
	wget http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz
	tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
	cd ..
	```

## Trained Models ##
  The following trained models can be found at [http://datasets.lids.mit.edu/fastdepth/results/](http://datasets.lids.mit.edu/fastdepth/results/).
  - MobileNet-NNConv5
  - MobileNet-NNConv5(depthwise)
  - MobileNet-NNConv5(depthwise), with additive skip connections
  - **MobileNet-NNConv5(depthwise), with additive skip connections, pruned**

Our final model is `mobilenet-nnconv5-skipadd-pruned`, i.e. a MobileNet-NNConv5 architecture with depthwise separable layers in the decoder, with additive skip connections between the encoder and decoder, and after network pruning using [NetAdapt](https://arxiv.org/pdf/1804.03230.pdf). The other models are offered to provide insight into our approach.

When downloading, save models to a `results` folder outside the repo directory:
  ```bash
  mkdir results; cd results
  wget -r -np -nH --cut-dirs=2 --reject "index.html*" http://datasets.lids.mit.edu/fastdepth/results/
  cd ..
  ```
### Pretrained MobileNet ###

The model file for the pretrained MobileNet used in our model definition can be downloaded from [http://datasets.lids.mit.edu/fastdepth/imagenet/](http://datasets.lids.mit.edu/fastdepth/imagenet/).

## Evaluation ##

This step requires a valid PyTorch installation and a saved copy of the NYU Depth v2 dataset. It is meant to be performed on a host machine with a CUDA GPU, not on an embedded platform. Deployment on an embedded device is discussed in the [next section](#deployment).

To evaluate a model, navigate to the repo directory and run:

```bash
python3 main.py --evaluate [path_to_trained_model]
```

Note: This evaluation code was sourced and modified from [here](https://github.com/fangchangma/sparse-to-dense.pytorch).

## Deployment ##

We use the [TVM compiler stack](https://tvm.ai/) to compile trained models for **deployment on an NVIDIA Jetson TX2**. Models are cross-compiled on a host machine and then deployed on the TX2. The `tvm-compile/tuning` folder in this repo contains the results of our [auto-tuning](https://docs.tvm.ai/tutorials/index.html#auto-tuning) the layers within our models for both the TX2 GPU and CPU. These can be used during the compilation process to achieve low model runtimes on the TX2. Outputs of TVM compilation for our trained models can be found at [http://datasets.lids.mit.edu/fastdepth/results/tvm_compiled/](http://datasets.lids.mit.edu/fastdepth/results/tvm_compiled/).

On the TX2, download the trained models as explained above in the section [Trained Models](#trained-models). The compiled model files should be located in `results/tvm_compiled`.

### Installing the TVM Runtime ####

Deployment requires building the TVM runtime code on the target embedded device (that will be used solely for running a trained and compiled model). The following instructions are taken from [this TVM tutorial](https://docs.tvm.ai/tutorials/cross_compilation_and_rpc.html#build-tvm-runtime-on-device) and have been tested on a **TX2 with CUDA-8.0 and LLVM-4.0 installed**.

First, clone the TVM repo and modify config file:
```bash
git clone --recursive https://github.com/dmlc/tvm
cd tvm
git reset --hard ab4946c8b80da510a5a518dca066d8159473345f
git submodule update --init
cp cmake/config.cmake .
```
Make the following edits to the `config.cmake` file:
```cmake
set(USE_CUDA OFF) -> set(USE_CUDA [path_to_cuda]) # e.g. /usr/local/cuda-8.0/
set(USE_LLVM OFF) -> set(USE_LLVM [path_to_llvm-config]) # e.g. /usr/lib/llvm-4.0/bin/llvm-config
```

Then build the runtime:
```bash
make runtime -j2
```
Finally, update the `PYTHONPATH` environment variable:
```bash
export PYTHONPATH=$PYTHONPATH:~/tvm/python
```
### Running a Compiled Model ####

To run a compiled model on the device, navigate to the `deploy` folder and run:

```bash
python3 tx2_run_tvm.py --input-fp [path_to_input_npy_file] --output-fp [path_to_output_npy_file] --model-dir [path_to_folder_with_tvm_compiled_model_files]
```

Note that when running a model compiled for the GPU, a `cuda` argument must be specified. For instance:

```bash
python3 tx2_run_tvm.py --input-fp data/rgb.npy --output-fp data/pred.npy --model-dir ../../results/tvm_compiled/tx2_cpu_mobilenet_nnconv5dw_skipadd_pruned/
python3 tx2_run_tvm.py --input-fp data/rgb.npy --output-fp data/pred.npy --model-dir ../../results/tvm_compiled/tx2_gpu_mobilenet_nnconv5dw_skipadd_pruned/ --cuda True
```

Example RGB input, ground truth, and model prediction data (as numpy arrays) is provided in the `data` folder. To convert the `.npy` files into `.png` format, navigate into `data` and run `python3 visualize.py`.

### Measuring Power Consumption ###

On the TX2, power consumption on the main VDD_IN rail can be measured by running the following command:

```bash
cat /sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power0_input
```

## Results

Comparison against prior work. Runtimes were measured on an NVIDIA Jetson TX2 in max-N mode.

| on NYU Depth v2     |  Input Size  |  MACs [G]  | RMSE | delta1 | CPU [ms] | GPU [ms] |
|---------------------------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:--:|
| [Eigen et al. [NIPS 2014]](https://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf)                | 228×304 | 2.06 | 0.907 | 0.611 | 300 | 23 |
| [Eigen et al. [ICCV 2015]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf) (with AlexNet) | 228×304 | 8.39 | 0.753 | 0.697 | 1400 | 96 |
| [Eigen et al. [ICCV 2015]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf) (with VGG)     | 228×304 | 23.4 | 0.641 | 0.769 | 2800 | 195 |
| [Laina et al. [3DV 2016]](https://arxiv.org/pdf/1606.00373.pdf) (with UpConv)   | 228×304 | 22.9 | 0.604 | 0.789 | 2400 | 237 |
| [Laina et al. [3DV 2016]](https://arxiv.org/pdf/1606.00373.pdf) (with UpProj)   | 228×304 | 42.7 | **0.573** | **0.811** | 3900 | 319 |
| [Xian et al. [CVPR 2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_Monocular_Relative_Depth_CVPR_2018_paper.pdf) (with UpProj)   | 384×384 | 61.8 | 0.660 | 0.781 | 4400 | 283 |
| This Work                                   | 224×224 | **0.37** | 0.604 | 0.771 | **37** | **5.6** |

"This Work" refers to MobileNet-NNConv5(depthwise), with additive skip connections, pruned.

<p float="left">
<img src="img/acc_fps_gpu.png" alt="photo not available" width="375">
<img src="img/acc_fps_cpu.png" alt="photo not available" width="375">
</p>

## Citation
If you reference our work, please consider citing the following:

	@inproceedings{icra_2019_fastdepth,
		author      = {{Wofk, Diana and Ma, Fangchang and Yang, Tien-Ju and Karaman, Sertac and Sze, Vivienne}},
		title       = {{FastDepth: Fast Monocular Depth Estimation on Embedded Systems}},
		booktitle   = {{IEEE International Conference on Robotics and Automation (ICRA)}},
		year        = {{2019}}
	}
