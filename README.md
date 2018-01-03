ALI-MNIST
===
PyTorch implementation of [Adversarially Learned Inference](https://arxiv.org/abs/1606.00704)

### Usage
```
$ ./train.py [-h] [-g GPU] [-b N] [-e E] [--lr LR] [--decay D] [-z Z]

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU     set GPU id (default: -1)
  -b N, --batch-size N  input batch size for training (default: 100)
  -e E, --epochs E      how many epochs to train (default: 100)
  --lr LR               learning rate (default: 1e-4)
  --decay D             weight decay or L2 penalty (default: 1e-5)
  -z Z, --zdim Z        dimension of latent vector (default: 128)
```

### result (<= 200 epochs)
![ali-1to200.gif](https://github.com/vwrs/ali-mnist/blob/imgs/ali-1to200.gif)

