import math

import torch
from torch import nn
from torch.nn import functional
from torch.nn.parameter import Parameter

from lib.utils import LogExp
from lib.utils import trace


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.l = len(args.layer)
        self.device = args.device
        self.inet = nn.ModuleList()
        darray = [args.d] + args.layer
        for i in range(self.l - 1):
            self.inet.append(nn.Linear(darray[i], darray[i + 1]))
        self.mu = nn.Linear(darray[self.l - 1], darray[self.l])
        self.sigma = nn.Linear(darray[self.l - 1], darray[self.l])
        self.gnet = nn.ModuleList()
        for i in range(self.l):
            self.gnet.append(nn.Linear(darray[self.l - i], darray[self.l - i - 1]))

    def encode(self, x):
        h = x
        for i in range(self.l - 1):
            h = functional.relu(self.inet[i](h))
        return self.mu(h), self.sigma(h)

    def decode(self, z):
        h = z
        for i in range(self.l - 1):
            h = functional.relu(self.gnet[i](h))
        return self.gnet[self.l - 1](h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class LVSM(nn.Module):
    def __init__(self, R, args):
        super(LVSM, self).__init__()
        self.alpha = args.alpha
        self.c = args.c
        self.d = args.d
        self.m, self.n = R.shape
        self.pi = torch.rand(self.c, args.m)
        self.pi /= torch.sum(self.pi, 0)
        self.pi = Parameter(self.pi)
        self.W = Parameter(torch.rand(self.c, args.d))
        self.logexp = LogExp.apply
        # nn.init.normal_(self.W.data, 0, 0.01)

    def score(self, pX, nX, pV, nV):
        # nu = torch.sum(self.R[u]).item()
        # neighbor = math.pow(nu, self.alpha)
        px = torch.sum(pX, dim=0)
        pv = torch.sum(pV, dim=0)
        pos = pX * (px - pX)
        pos = torch.sum(pos.unsqueeze(0) * self.W.unsqueeze(1), -1)
        pos += torch.sum(pV * (pv - pV), -1)
        # pos /= neighbor

        # nx = torch.sum(nX, dim=0)
        # nv = torch.sum(nV, dim=0)
        neg = px * nX
        neg = torch.sum(neg.unsqueeze(0) * self.W.unsqueeze(1), -1)
        neg += pv @ nV.t()
        # neg /= neighbor

        alpha = torch.cat([pos, neg], -1)
        return torch.sum(pos, -1) - torch.sum(self.logexp(alpha), -1)

    def maximum(self, pX, nX, pV, nV, u):
        pi = self.pi[:, u].detach()
        score = self.score(pX, nX, pV, nV)
        # score = torch.sum(pos, -1)
        return -pi @ score

    def expect(self, pX, pV, u):
        # nu = torch.sum(self.R[u]).item()
        # neighbor = math.pow(nu, self.alpha)
        px = torch.sum(pX, dim=0)
        pv = torch.sum(pV, dim=0)
        pos = pX * (px - pX)
        pos = torch.sum(pos.unsqueeze(0) * self.W.unsqueeze(1), -1)
        pos += torch.sum(pV * (pv - pV), -1)
        # pos /= neighbor
        gamma = torch.sum(pos, -1) - torch.sum(self.logexp(pos), -1)
        pi = functional.softmax(gamma - 1, dim=0)
        self.pi.data[:, u] = functional.one_hot(torch.argmax(pi), self.c)

    def forward(self, pX, nX, pV, nV, u, expect=True):
        return self.expect(pX, pV, u) if expect else self.maximum(pX, nX, pV, nV, u)

    def predict(self, x, tX, v, tV, u):
        w = self.pi[:, u].t() @ self.W
        score = (x * w) @ tX.t() + v @ tV.t()
        return score

    # def predict_all(self, R, rX, tX, rV, tV):
    #     w = self.pi.t() @ self.W
    #     return R @ rX * w @ tX.t() + R @ rV @ tV.t()


class LSM(nn.Module):
    def __init__(self, R, args):
        super(LSM, self).__init__()
        self.alpha = args.alpha
        self.c = args.c
        self.R = R
        self.m, self.n = R.shape
        self.pi = Parameter(torch.rand(self.c, args.m) / self.c)
        self.W = Parameter(torch.rand(self.c, args.d))
        self.logexp = LogExp.apply

    def score(self, pX, nX, u):
        nu = torch.sum(self.R[u]).item()
        neighbor = math.pow(nu, self.alpha)
        px = torch.sum(pX, dim=0)
        pos = pX * (px - pX)
        pos = torch.sum(pos.unsqueeze(0) * self.W.unsqueeze(1), -1)
        pos /= neighbor

        nx = torch.sum(nX, dim=0)
        neg = nx * nX
        neg = torch.sum(neg.unsqueeze(0) * self.W.unsqueeze(1), -1)
        neg /= neighbor

        alpha = torch.cat([pos, neg], -1) / neighbor
        return torch.sum(pos, -1) - torch.sum(self.logexp(alpha), -1)

    def expect(self, pX, nX, u):
        gamma = self.score(pX, nX, u)
        self.pi.data[:, u] = functional.softmax(gamma - 1, 0)

    def maximum(self, pX, nX, u):
        pi = self.pi[:, u].detach()
        score = self.score(pX, nX, u)
        return -pi @ score

    def forward(self, pX, nX, u, expect=True):
        return self.expect(pX, nX, u) if expect else self.maximum(pX, nX, u)

    def predict(self, x, tX, u):
        w = self.pi[:, u].t() @ self.W
        score = (x * w) @ tX.t()
        return score
