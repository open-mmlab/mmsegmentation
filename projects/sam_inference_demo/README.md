# Introducing the Segment Anything Model (SAM) Inference Demo!

Welcome to the Segment Anything (SA) Inference Demo, a user-friendly implementation based on the original Segment Anything project. Our demo allows you to experience the power and versatility of the Segment Anything Model (SAM) through an easy-to-use API.

With this inference demo, you can explore the capabilities of the Segment Anything Model and witness its effectiveness in various tasks and image distributions. For more information on the original project, dataset, and model, please visit the official website at https://segment-anything.com.

### Prerequisites

- Python 3.10
- PyTorch 1.13
- MMEngine >= v0.7.2
- MMCV >= v2.0.0

### Installation

We assume that you have already installed PyTorch. If not, please follow the instructions on the [PyTorch website](https://pytorch.org/).

**1. Install MMEngine & MMCV**

```shell
pip install openmim
mim install mmengine
mim install 'mmcv>=2.0.0'
```

**2. Install MMPretrain**

```shell
pip install git+https://github.com/open-mmlab/mmpretrain.git@dev
```

**3. Install MMSegmentation**

```shell
pip install mmsegmentation
```

### Usage

Open the `sam_image_demo.ipynb` notebook and follow the instructions to run the demo.
