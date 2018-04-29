import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
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

        # return epoch_loss


# learner = Learner(Object, Learning, )
# learner.learn(200)
