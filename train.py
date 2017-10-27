#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import argparse
import chainer
from chainer import training
from chainer import iterators, optimizers, serializers
from chainer import cuda
from chainer.training import extensions
import dataset
import network

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--epoch', '-e', type=int, default=3)
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--inf', type=int, default=10)
    parser.add_argument('--outf', type=int, default=10)
    parser.add_argument('--batch', '-b', type=int, default=8)
    args = parser.parse_args()

    train = dataset.MovingMnistDataset(0, 7000, args.inf, args.outf)
    train_iter = iterators.SerialIterator(train, batch_size=args.batch, shuffle=True)
    test = dataset.MovingMnistDataset(7000, 10000, args.inf, args.outf)
    test_iter = iterators.SerialIterator(test, batch_size=args.batch, repeat=False, shuffle=False)

    model = network.MovingMnistNetwork(sz=[128,64,64], n=2)

    if args.model != None:
        print( "loading model from " + args.model )
        serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        model.to_gpu()

    opt = optimizers.Adam(alpha=args.lr)
    opt.setup(model)

    if args.opt != None:
        print( "loading opt from " + args.opt )
        serializers.load_npz(args.opt, opt)

    updater = training.StandardUpdater(train_iter, opt, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    trainer.run()

    modelname = "./results/model"
    print( "saving model to " + modelname )
    serializers.save_npz(modelname, model)

    optname = "./results/opt"
    print( "saving opt to " + optname )
    serializers.save_npz(optname, opt)

if __name__ == '__main__':
    train()
