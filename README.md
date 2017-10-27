# ConvLSTM

Convolutional LSTM implemented with chainer

`python 3.5.2` + `chainer 3.0.0`

## Getting dataset

```
$ ./get-moving-mnist.sh
```

## Training

```
$ python3 ./train.py -g 0 --epoch 10 --inf 3 --outf 3 --batch 16
```

## Generating

```
$ mkdir img
$ python3 ./generate.py --model results/model --id 7000 --inf 3 --outf 3
```

Then, the images are generated in `img/`.

## Gallery

![7000_cap](https://github.com/joisino/ConvLSTM/blob/master/imgs/7000_cap.png)

![7001](https://github.com/joisino/ConvLSTM/blob/master/imgs/7001.png)

![7002](https://github.com/joisino/ConvLSTM/blob/master/imgs/7002.png)

![7003](https://github.com/joisino/ConvLSTM/blob/master/imgs/7003.png)

![7004](https://github.com/joisino/ConvLSTM/blob/master/imgs/7004.png)
