# image_classification


## Installation

### Reauirements

- Linux
- Python 3.7
- CUDA 10.0
- PyTorch 1.3

### Install

a. Create a conda virtual environment and activate it.

```shell
conda create -n smoke_classification python=3.7 -y
conda activate smoke_classification
```
b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch=1.3 torchvision cudatoolkit=10.0 -c pytorch
```

c. Clone the image_classification repository.

```shell
git clone https://github.com/JOOCHANN/image_classification.git
cd image_classification
```

d. Install image_classification

```shell
pip install pandas
pip install sklearn
```

### DEMO

```shell
python demo.py --load_model_path='' --out_file='' --demo_data_path=''
```

### TEST

```shell
python main.py --mode 'test' --load_model_path='' --out_file='' --test_data_path=''
```
