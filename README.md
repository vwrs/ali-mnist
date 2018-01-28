ALI-MNIST
===
Simple PyTorch implementation of [Adversarially Learned Inference](https://arxiv.org/abs/1606.00704)

### Usage
```
$ ./train.py [-h] [-g GPU] [-b N] [-e E] [--lr-g LR] [--lr-d LR] [--decay D] [-z Z]

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU     set GPU id (default: -1)
  -b N, --batch-size N  input batch size for training (default: 32)
  -e E, --epochs E      how many epochs to train (default: 100)
  --lr-g LR             initial ADAM learning rate of G (default: 2e-4)
  --lr-d LR             initial ADAM learning rate of D (default: 1e-5)
  --decay D             weight decay or L2 penalty (default: 0)
  -z Z, --zdim Z        dimension of latent vector (default: 128)
```

### result (<= 200 epochs)
![ali-1to200.gif](https://github.com/vwrs/ali-mnist/blob/imgs/ali-1to200.gif)

