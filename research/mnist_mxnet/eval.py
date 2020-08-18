# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
from __future__ import print_function

import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import os
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn

# Parse CLI arguments

parser = argparse.ArgumentParser(description='MXNet Gluon MNIST Example')
parser.add_argument('--data_path', default='./data', help='the dir to download datasets')
parser.add_argument('--batch-size', type=int, default=100,
                    help='batch size for training and testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='Train on GPU with CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--checkpoint_path', default='./data', help='the dir to save model')
parser.add_argument('--output_path', default='./data', help='the dir to save model')
opt = parser.parse_args()


# define network

net = nn.Sequential()
net.add(nn.Dense(128, activation='relu'))
net.add(nn.Dense(64, activation='relu'))
net.add(nn.Dense(10))

# data

def transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32)/255
    return data, label

val_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST(opt.data_path, train=False).transform(transformer),
    batch_size=opt.batch_size, shuffle=False)

# train

def test(ctx):
    metric = mx.gluon.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])

    return metric.get()

def main(ctx):
    # Collect all parameters from net and its children, then initialize them.
    checkpoint_file = os.path.join(opt.checkpoint_path, "mnist.params")
    net.load_parameters(checkpoint_file, ctx=ctx)
    name, val_acc = test(ctx)
    print('Validation: %s=%f'%( name, val_acc))


if __name__ == '__main__':
    if opt.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()
    main(ctx)
