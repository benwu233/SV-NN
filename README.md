# SV-NN
Code repository of the paper:

Ben Wu, Keru Wu, Jian Kang. (2025). Bayesian Scalar-on-Image Regression with a Spatially Varying Single-layer Neural Network Prior. *Journal of Machine Learning Research*. In Press.

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

Then install R package 'BayesGPfit':

```
python install_BayesGPfit.py
```

## Experiments

### MNIST: 
The pair number follows the order of digit pairs in the paper, e.g.

5 vs. 7, full data:

```
python MNIST_acc.py -pair 2 -all True
```

5 vs. 7, n=10

```
python MNIST_acc.py -pair 2 -all False -size 10
```

### Fashion MNIST: 

```
python FashionMNIST_acc.py -pair 2
python FashionMNIST_acc.py -pair 2 -all False -size 10
```

### Neuroimaging

```
python neuroimaging_acc.py -img 2bk-0bk -split random
python neuroimaging_acc.py -img 2bk-baseline -split single
```




