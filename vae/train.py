"""
Train a VAE model using saved images in a folder
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from .model import VAE, create_optimizer, loss_function
from .dataset import build_dataset

def train(args):
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # get dataset
    train_dataset, test_dataset = build_dataset(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=16,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=16,
        drop_last=True,
    )

    # create model
    vae = VAE(zdim=args.zdim, training=True).to(device)

    # create optimizer
    optimizer = create_optimizer(args, vae)

    # loss list
    training_loss_list = []

    # train loop
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            recon_batch, mean, log_var = vae(batch)
            loss = loss_function(recon_batch, batch, mean, log_var, args.beta)

            loss_value = loss.item()
            training_loss_list.append(loss_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("epoch:", epoch, "train loss:", loss_value)

        # save checkpoint
        if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            path = 'logs_noaug_mini/train_epoch_' + str(epoch) + '.pth'
            torch.save({
                'epoch': epoch,
                'model': vae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss_list': training_loss_list
            }, path)

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Path to a folder containing images for training', type=str,
                    default='./recorded_data/')
    parser.add_argument('--zdim', help='Latent space dimension', type=int, default=512)
    parser.add_argument('--seed', help='Random generator seed', type=int, default=42)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=64)
    parser.add_argument('--learning-rate', help='Learning rate', type=float, default=1e-4)
    parser.add_argument('--kl-tolerance', help='KL tolerance (to cap KL loss)', type=float, default=0.5)
    parser.add_argument('--beta', help='Weight for kl loss', type=float, default=1.0)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--save-ckpt-freq', help='Save checkpoint frequency', type=int, default=10)
    args = parser.parse_args()

    # train the VAE model
    train(args)