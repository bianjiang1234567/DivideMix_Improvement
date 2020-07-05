# DivideMix: Learning with Noisy Labels as Semi-supervised Learning

# Two Improvements

## DivideMix is more suitable for large noise samples, and the performance with less noise can be improved by using the equality strategy

<b>Experiments:</b>
First, please create a folder named <i>checkpoint</i> to store the results.\
<code>mkdir checkpoint</code>

### To compare, use the following commands

```bash
$ python Train_cifar.py --lr 0.001 --r 0.2 --net CNN_small --opt adam --num_epochs 300
$ python Train_cifar.py --lr 0.001 --r 0.2 --equal True --net CNN_small --opt adam --num_epochs 300
```
### Results

<div align="center">
<img src=assets/README-745fc89d.jpg width = 45% height = 50%/>
<img src=assets/README-0e109386.jpg width = 45% height = 50%/>
</div>
<div align="center">
DivideMix   　　　　        DivideMix With Equality Strategy
</div>

## DivideMix cannot handle the case of category imbalance. We set the category ratio to be 9：1, and integrate Label-distribution-aware Margin Loss to improve the performance

### To compare, use the following commands

```bash
$ python Train_cifar.py --lr 0.001 --r 0.1 --noise_mode asym_two_unbalanced_classes --net CNN_small --opt adam --num_epochs 300 --num_class 2
$ python Train_cifar.py --lr 0.001 --r 0.1 --noise_mode asym_two_unbalanced_classes --net CNN_small --opt adam --num_epochs 300 --num_class 2 --LDAM_DRW True
```

### Results

<div align="center">
<img src=assets/README-ce9cea91.jpg width = 45% height = 50%/>
<img src=assets/README-c93d020b.jpg width = 45% height = 50%/>
</div>
<div align="center">
DivideMix   　　　　        DivideMix With Label-distribution-aware Margin Loss
</div>
