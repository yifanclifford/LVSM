import argparse

import numpy as np
import torch
import yaml
from scipy import io
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.utils import Evaluator
from model.vae import VAE


def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train():
    vae.train()
    for epoch in range(args.maxiter):
        train_loss = 0
        for batch_idx, idx in enumerate(tqdm(loader)):
            optimizer.zero_grad()
            f_batch = torch.from_numpy(X[idx]).to(args.device)
            recon_batch, mu, logvar = vae(f_batch)
            loss = vae_loss(recon_batch, f_batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(loader.dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='running on the GPU', action='store_true')
    parser.add_argument('--save', help='whether to save the model', action='store_true')
    parser.add_argument('--load', help='whether to load the model', action='store_true')
    parser.add_argument('--layer', nargs='+', help='number of neurals in each layer', type=int, default=[20])
    parser.add_argument('--loss', help='loss function, currently support binary and mse', choices=['log', 'mse'],
                        default='mse')
    parser.add_argument('--batch', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
    parser.add_argument('-b', '--beta', help='parameter beta', type=float, default=.1)
    parser.add_argument('-m', '--maxiter', help='max number of iteration', type=int, default=5)

    args = parser.parse_args()
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    with open('config') as file:
        params = yaml.load(file, Loader=yaml.Loader)
        layer = '_'.join([str(l) for l in args.layer])

    path = '{}/{}/{}'.format(params['dataset_dir'], params['dataset'], params['feature'])
    print('feature data path: ' + path)
    X = io.mmread(path).A.astype('float32')
    args.n, args.d = X.shape

    # X = normalize(X, norm='l2', axis=0)

    vae = VAE(args)
    if args.load:
        path = '{}/model/vae_{}_{}'.format(params['dataset_dir'], params['dataset'], layer)
        vae.load_state_dict(torch.load(path))

    vae.to(args.device)

    # psm = PSM(torch.from_numpy(R.astype('float32')).to(args.device), args).to(args.device)

    loader = DataLoader(np.arange(args.n), batch_size=1, shuffle=True)
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)

    evaluator = Evaluator({'recall', 'dcg_cut'})
    # variational()
    # maximum()
    # evaluate()
    train()

    if args.save:
        vae.cpu()
        path = '{}/model/vae_{}_{}'.format(params['dataset_dir'], params['dataset'], layer)
        torch.save(vae.state_dict(), path)
