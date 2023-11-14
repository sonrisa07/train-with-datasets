<h3 align="center">Train With Datasets</h3>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![GitHub Issues](https://img.shields.io/github/issues/sonrisa07/train-with-datasets.svg)](https://github.com/sonrisa07/train-with-datasets/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/sonrisa07/train-with-datasets.svg)](https://github.com/sonrisa07/train-with-datasets/pulls)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/license/mit/)

</div>

---

<p align="center"> within this repository, you'll find a diverse array of datasets, all harnessed to refine and train our models.
    <br> 
</p>

## 📝 Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)
- [Deployment](#deployment)

## 🧐 About <a name = "about"></a>
During I learn deep learning, I am about to record the datasets which I will have encountered.

Simultaneously, I will train and test the datasets by various suitable models as much as possible. I will have learned and add some comments in some crucial places.


## 🏁 Getting Started <a name = "getting_started"></a>
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites
You must install the relevant dependency packages.
- [Python](https://www.python.org) - programming language
- [PyTorch](http://pytorch.org) - deep learning framework
  - `pip3 install torch torchvision torchaudio`
- [NumPy](https://numpy.org) - numerical calculation tools
  - `pip install numpy`
- [Matplotlib](https://matplotlib.org) - python drawing library
  - `pip install matplotlib`
- [rich](https://github.com/Textualize/rich) - a Python library for rich text and beautiful formatting in the terminal
  - `pip install rich`

Simultaneously, every dataset needs to be downloaded if you plan to use it. Please ensure you can get online.

### ⛏️ Installing
After installing related dependencies, you simply have to clone this repository locally.

use ssh
```shell
git clone git@github.com:sonrisa07/train-with-datasets.git
```
use https
```shell
git clone https://github.com/sonrisa07/train-with-datasets.git
```

## 🎈 Usage <a name="usage"></a>
Please ensure that you are currently in the repository directory.

There are numerous models, each containing many executable python files using different datasets. Then, pick one of them and begin executing it.

Next, we use the vgg model and CIFAR-10 dataset as an example. The python file is `./models/vgg/cifar10.py`.

Firstly, you can check the optional command parameters.

```shell
python /models/vgg/cifar10.py --help
```

For instance, you can customize parameters such as epochs and learning rate.

```shell
python /models/vgg/cifar10.py -e 20 --lr 1e-3
```

You can also opt for various models to apply to this dataset.
```shell
python /models/vgg/cifar10.py -m VGG16
```
>⚠️ The models folders include various models, but it's important to note that not all models are suitable for every dataset.

Hence, you should select a fitting model for the dataset. At the outset of each Python file for the dataset, optional model parameters are provided for your use. Models outside the designated range are not compatible.

Certainly, you can directly adjust the parameters in the dataset python file.

The program will save the model's parameter file from the best performance on the validation set to the local storage, and you can also modify the path.

## 🚀 Deployment <a name = "deployment"></a>
During development, I've worked to minimize the coupling between models and datasets. 

This allows you to seamlessly integrate models into your project, and you can also fine-tune parameters in the code to meet your objectives.

I trust in your capability.
