import numpy as np
import json
import pandas as pd
import math
from flask import Flask, request
import torch
import torch.nn as nn
import time


# import hello.py as


class Model(nn.Module,):
    def __init__(self, obj, bounds, init=False):
        super(Model, self).__init__()
        self.coor = nn.Parameter(torch.rand(2))

        self.register_parameter('coordinate', self.coor)
        # self.coor[0] = (bounds[1] - bounds[0]) * self.coor[0] + bounds[0]
        # self.coor[1] = (bounds[2] - bounds[3]) * self.coor[1] + bounds[2]
        self.obj = obj

    def forward(self):
        # f = torch.sum(torch.pow(self.coor, 2))

        x = self.coor[0]
        y = self.coor[1]

        if self.obj == 'flower':
            f = (x * x) + (y * y) + x * torch.sin(y) + y * torch.sin(x)
            return f

        if self.obj == 'himmelblau':
            f = torch.pow(x * x - 11, 2) + torch.pow(x + y * y - 7, 2)
            return f

        if self.obj == 'banana':
            f = torch.pow(1 - x, 2) + 100 * torch.pow(y - x * x, 2)
            return f

        if self.obj == 'matyas':
            f = 0.26 * (x * x + y * y) + 0.48 * x * y
            return f

        # def getObjective(s):
        #     return lambda x, y: eval(s)

        # evals = getObjective(self.obj)
        f = eval(self.obj)
        return f


class Learner(object):
    def __init__(self, obj='x*y', bounds=[-6, 6, -6, 6]):
        self.coordinates = []
        self.time = []
        self.model = Model(obj, bounds)
        self.lam = 1e-5

    def learn(self, opt='lbfgs', epochs=50, lam=1e-3, rate=1e-3):

        if opt == 'lbfgs':

            def fun_closure():
                loss = self.model.forward()
                optimizer.zero_grad()
                loss.backward()
                cpu_time = time.clock()
                self.coordinates.append(self.model.coor.data.numpy())
                self.time.append(cpu_time)
                return loss

            optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                lr=rate)
            for epoch in range(epochs):
                optimizer.step(fun_closure)

        else:
            # set optimizer
            if opt == 'GD':
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=rate, weight_decay=lam)

            if opt == 'adam':
                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=rate, weight_decay=lam)

            if opt == 'adagrad':
                optimizer = torch.optim.Adagrad(
                    self.model.parameters(),
                    lr=rate, weight_decay=lam)

            if opt == 'adadelta':
                optimizer = torch.optim.Adadelta(
                    self.model.parameters(),
                    lr=rate, weight_decay=lam)

            if opt == 'rmsprop':
                optimizer = torch.optim.RMSprop(
                    self.model.parameters(),
                    lr=rate, weight_decay=lam)

            if opt == 'GDM':
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=rate, weight_decay=lam, momentum=0.01)

            if opt == 'rmspropM':
                optimizer = torch.optim.RMSprop(
                    self.model.parameters(),
                    lr=rate, weight_decay=lam, momentum=0.01)

            for epoch in range(epochs):
                loss = self.model.forward()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cpu_time = time.clock()

                self.coordinates.append(self.model.coor.data.numpy().tolist())
                self.time.append(cpu_time)


# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')


@app.route('/')
def homepage():
    return app.send_static_file('index.html')


def scaledValue(width, height, x1, x2, y1, y2, f):
    X_scale = (x2 - x1) / width * 1.0
    Y_scale = (y2 - y1) / height * 1.0

    arr = []
    for y in np.arange(y1, y2, Y_scale):
        for x in np.arange(x1, x2, X_scale):
            value = f(x, y)
            arr.append(value)

    # print(width, height)
    # print(len(arr))
    return arr


def getObjective(s):
    return lambda x, y: eval(s)


def flower(x, y):
    return x * x + y * y + x * math.sin(y) + y * math.sin(x)


def matyas(x, y):
    return 0.26 * (x * x + y * y) + 0.48 * x * y


def banana(x, y):
    return math.pow(1 - x, 2) + 100 * math.pow(y - x * x, 2)


def himmelblau(x, y):
    return math.pow(x * x - 11, 2) + math.pow(x + y * y - 7, 2)


# Soliciting effective training examples from bootstrap
@app.route('/training', methods=['POST'])
def training():

    data = json.loads(request.data)
    learning_rates = data["rate"]
    optimizers = data["opt"]
    objective = data["obj"]
    # epoch = data["epoch"]
    decay_dates = data["reg"]
    width = data["width"]
    height = data["height"]
    customize = data["customize"]
    print('--------------------')
    print(data)
    print(data['pos'])
    print('--------------------')
    # [x1, x2] = data["X"]
    [x1, x2] = [-6, 6]
    # [y1, y2] = data["Y"]
    [y1, y2] = [-6, 6]

    f = flower
    if customize:
        f = getObjective(objective)
    else:
        if objective == 'flower':
            f = flower
        elif objective == 'banana':
            f = banana
        elif objective == 'himmelblau':
            f = himmelblau
        else:
            f = matyas

    values = scaledValue(width, height, x1, x2, y1, y2, f)
    res = {}
    res["values"] = values

    for opt in optimizers:
        for rate in learning_rates:
            for reg in decay_dates:
                key = opt + '-' + rate + '-' + reg
                learner = Learner(objective, [x1, x2, y1, y2])
                learner.learn(opt=opt, lam=float(reg), rate=float(rate))
                res[key] = learner.coordinates

    return json.dumps({'res': res}), 200, {'ContentType': 'application/json'}


app.run(debug=True)
