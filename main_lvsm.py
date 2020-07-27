import argparse
import os
import shutil
from time import time

import numpy as np
import torch
import yaml
from scipy import io
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.metric import metric_recall, metric_dcg
from lib.utils import sample_negative, trace
from model.fsm import LVSM, VAE


def vae_loss(recon_x, x, mu, logvar):
    BCE = functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def evaluate():
    print('evaluate performance')
    start = time()
    rV, _ = vae.encode(rX)
    tV, _ = vae.encode(tX)
    rV = rV.detach()
    tV = tV.detach()
    lvsm.eval()
    recall = dcg = 0
    for user in tqdm(test_loader):
        ru = torch.from_numpy(R[user].astype('float32')).to(args.device)
        x = ru @ rX
        v = ru @ rV
        _, rec_items = torch.topk(lvsm.predict(x, tX, v, tV, user).detach(), 10)
        rec_items = rec_items.cpu().numpy()
        for idx, u in enumerate(user):
            recall += metric_recall(rec_items[idx], test[u.item()], [10])[0]
            dcg += metric_dcg(rec_items[idx], test[u.item()], [10])[0]

    return recall / len(test_user), dcg / len(test_user), time() - start


def expect():
    print('variational E-step')
    start = time()
    for u in tqdm(train_loader):
        u = u.item()
        pos_items = np.flatnonzero(R[u])
        neg_items = np.flatnonzero(1 - R[u])
        # l = len(pos_items)

        neg_items = torch.from_numpy(neg_items).to(args.device)
        pos_items = torch.from_numpy(pos_items).to(args.device)

        pX = rX[pos_items]
        nX = rX[neg_items]

        pV, _ = vae.encode(pX)
        nV, _ = vae.encode(nX)

        lvsm(pX, nX, pV, nV, u)
    return time() - start


def maximum():
    print('variational M-step')
    train_loss = 0
    start = time()
    for u in tqdm(train_loader):
        optimizer.zero_grad()
        u = u.item()
        pos_items = np.flatnonzero(R[u])
        # l = len(pos_items)

        neg_items = torch.from_numpy(sample_negative(pos_items, args.n, params['num_negative'])).to(args.device)
        pos_items = torch.from_numpy(pos_items).to(args.device)

        pX = rX[pos_items]
        nX = rX[neg_items]

        recon, pV, sigma = vae(pX)
        loss = args.beta * vae_loss(recon, pX, pV, sigma)
        recon, nV, sigma = vae(nX)
        loss += args.beta * vae_loss(recon, nX, nV, sigma)
        loss += lvsm(pX, nX, pV, nV, u, False)
        loss += args.beta * trace(lvsm.W)
        # print(loss)
        # if torch.isinf(loss):
        #     print(len(pos_items))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss, time() - start


def train():
    for epoch in range(args.maxiter):
        lvsm.train()
        t1 = expect()
        print(lvsm.pi[:, 1:10])
        loss, t2 = maximum()
        recall, dcg, t3 = evaluate()

        print('====> Epoch: {}, Average loss: {:.4f}, '
              'E-step time: {}, M-step time: {}, '
              'Evaluation time: {}, Recall={}, DCG={}'
              .format(epoch, loss / args.m, t1, t2, t3, recall, dcg))

        path = '{}/{}/result/LVSM{}'.format(params['dataset_dir'], params['dataset'], args.task)
        with open(path, 'a') as file:
            file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(epoch, recall, dcg, t1, t2, t3))

        if args.save:
            lvsm.cpu()
            path = '{}/{}/LVSM{}/LVSM_{}'.format(params['dataset_dir'], params['dataset'], args.task, epoch)
            torch.save(lvsm.state_dict(), path)
            lvsm.to(args.device)

            vae.cpu()
            path = '{}/{}/LVSM{}/VAE_{}'.format(params['dataset_dir'], params['dataset'], args.task, epoch)
            torch.save(vae.state_dict(), path)
            vae.to(args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='running on the GPU', action='store_true')
    parser.add_argument('--save', help='whether to save each epoch', action='store_true')
    parser.add_argument('--pretrain', help='whether to pretrain VAE', action='store_true')
    parser.add_argument('--layer', nargs='+', help='number of neurons in each layer', type=int, default=[20])
    parser.add_argument('--loss', help='loss function, currently support binary and mse', choices=['log', 'mse'],
                        default='mse')
    parser.add_argument('--task', type=int, help='get the id of the task', default=0)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
    parser.add_argument('-a', '--alpha', help='parameter alpha', type=float, default=1)
    parser.add_argument('-b', '--beta', help='parameter beta', type=float, default=.1)
    parser.add_argument('-l', '--lamb', help='parameter lambda', type=float, default=.1)
    parser.add_argument('-c', help='number of clusters', type=int, default=1)
    parser.add_argument('-m', '--maxiter', help='max number of iteration', type=int, default=5)

    args = parser.parse_args()
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    with open('config') as file:
        params = yaml.load(file, Loader=yaml.Loader)
        layer = '_'.join([str(l) for l in args.layer])

    # print('dataset directory: ' + params['dataset_dir'])
    path = '{}/{}/cold/train.txt'.format(params['dataset_dir'], params['dataset'])
    print('train data path: ' + path)
    R = io.mmread(path).A.astype('int')
    path = '{}/{}/{}'.format(params['dataset_dir'], params['dataset'], params['feature'])
    print('feature data path: ' + path)
    X = io.mmread(path).A.astype('float32')
    # X = normalize(X, norm='l2', axis=0)

    path = '{}/{}/cold/valid.txt'.format(params['dataset_dir'], params['dataset'])
    print('valid path: {}'.format(path))
    T = io.mmread(path).A.astype('int')
    warm = np.flatnonzero(np.sum(R, axis=0) > 0).tolist()
    cold = np.flatnonzero(np.sum(T, axis=0) > 0).tolist()
    args.m = R.shape[0]
    args.n = len(warm)
    args.d = X.shape[1]
    R = R[:, warm]
    rX = torch.from_numpy(X[warm, :]).to(args.device)
    tX = torch.from_numpy(X[cold, :]).to(args.device)
    T = T[:, cold]
    test_user = np.flatnonzero(np.sum(T, axis=1))
    test = {u: {i for i in np.flatnonzero(T[u])} for u in test_user}

    vae = VAE(args)
    if args.pretrain:
        path = '{}/model/vae_{}_{}'.format(params['dataset_dir'], params['dataset'], layer)
        vae.load_state_dict(torch.load(path))
    vae.to(args.device)

    lvsm = LVSM(torch.from_numpy(R.astype('float32')).to(args.device), args).to(args.device)
    lvsm_params = sum(p.numel() for p in lvsm.parameters() if p.requires_grad)
    vae_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print('#params of LVSM={}, #param of VAE={}, #param total={}'.format(lvsm_params, vae_params,
                                                                         lvsm_params + vae_params))

    path = '{}/{}/LVSM{}'.format(params['dataset_dir'], params['dataset'], args.task)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    train_loader = DataLoader(np.arange(args.m), batch_size=1, shuffle=True)
    test_loader = DataLoader(test_user, batch_size=params['test_batch'], shuffle=True)
    optimizer = optim.Adam(list(vae.parameters()) + list(lvsm.parameters()), lr=args.lr)
    # optimizer = optim.Adam(psm.parameters(), lr=args.lr)
    # variational()
    # maximum()
    # evaluate()
    train()
