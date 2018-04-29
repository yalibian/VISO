import numpy as np
import json
import pandas as pd
from flask import Flask, request
import torch
import torch.nn as nn
# import hello.py as


# eval('./data/hello world')
class Model(nn.Module):
    # class Model(nn.Module, objectiveFunction=eval("./data")):
    def __init__(self, init=False):
        super(Model, self).__init__()
        self.coor = nn.Parameter(torch.randn(2))
        self.register_parameter('coordinate', self.coor)

    def forward(self):
        y = torch.sum(torch.pow(self.coor, 2))
        return y


class Learner(object):
    def __init__(self):
        self.coordinates = []
        self.losses = []
        self.model = Model()
        self.lam = 1e-5

    def learn(self, epochs=50):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=1e-2)

        # weight_decay=self.lam)
        for epoch in range(epochs):
            loss = self.model.forward()
            epoch_loss = loss.data.numpy()[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Iteration {}: Objective = {}".format(epoch, epoch_loss))

            self.losses.append(epoch_loss)
            self.coordinates.append(self.model.coor.data.numpy())


# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')


@app.route('/')
def homepage():
    return app.send_static_file('index.html')


# Soliciting effective training examples from bootstrap
@app.route('/training', methods=['POST'])
def training():
    # instances = state['instances']

    print("In Training")
    print(request.data)

    data = json.loads(request.data)
    rate = data["rate"]
    opt = data["opt"]
    obj = data["obj"]
    epoch = data["epoch"]
    print(rate)
    print(opt)
    print(obj)
    print(epoch)


    return json.dumps({"res": "hello world"}), 200, {'ContentType': 'application/json'}
    #-------------------------------------
    # Mi
    # arry1 = function getValues()




    # -------------------------------------
    # Yao
    res = {}
    for opt in data.opt:
        for rate in data.learning_rate:
            learner = Learner()
            learner.learn(data.epoch)
            res["learner"] = learner.coordinates




    # -------------------------------------
    # res["pos"].array = Mi

    return json.dumps({"res": "hello world"}), 200, {'ContentType': 'application/json'}


# if __name__ == '__main__':
app.run(debug=True)
