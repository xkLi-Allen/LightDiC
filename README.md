## LightDiC: A Simple yet Effective Approach for Large-scale Digraph Representation Learning

**Requirements**

Hardware environment: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz, NVIDIA GeForce RTX 3090 with 24GB memory.

Software environment: Ubuntu 18.04.6, Python 3.9, PyTorch 1.11.0 and CUDA 11.8.

1. Please refer to [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install the environments;
2. Run 'pip install -r requirements.txt' to download required packages;

**Training**

To train the model(s) in the paper

1. Please unzip xxx.zip to the current file directory location

2. Please refer to the configs folds to modify the hyperparameters 

   data_config.py - dataset loading

   model_config.py - model initialization  

   training_config.py - training stages

3. Open main.py to train digraph learning model.

    We provide CoraML/CiteSeer/WikiTalk dataset as example (Execute data set partitioning and processing).

    Meanwhile, you can personalize your settings (data/model/training)

    Run this command:

```python
  python main.py
```
