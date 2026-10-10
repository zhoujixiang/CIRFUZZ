#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)

    return gaussian_kernel


class KNN(nn.Module):
    def __init__(self, params, nclasses):
        super().__init__()
        print("*" * 80)
        print("Cleaning point-clouds with kNN post-processing")
        self.knn = params["knn"]
        self.search = params["search"]
        self.sigma = params["sigma"]
        self.cutoff = params["cutoff"]
        self.nclasses = nclasses
        print("kNN parameters:")
        print("knn:", self.knn)
        print("search:", self.search)
        print("sigma:", self.sigma)
        print("cutoff:", self.cutoff)
        print("nclasses:", self.nclasses)
        print("*" * 80)

    def forward(self, proj_range, unproj_range, proj_argmax, px, py):
        if proj_range.is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        H, W = proj_range.shape

        P = unproj_range.shape

        if (self.search % 2 == 0):
            raise ValueError("Nearest neighbor kernel must be odd number")

        pad = int((self.search - 1) / 2)

        proj_unfold_k_rang = F.unfold(proj_range[None, None, ...],
                                      kernel_size=(self.search, self.search),
                                      padding=(pad, pad))

        idx_list = py * W + px
        unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list]

        unproj_unfold_k_rang[unproj_unfold_k_rang < 0] = float("inf")

        center = int(((self.search * self.search) - 1) / 2)
        unproj_unfold_k_rang[:, center, :] = unproj_range

        k2_distances = torch.abs(unproj_unfold_k_rang - unproj_range)

        inv_gauss_k = (
                1 - get_gaussian_kernel(self.search, self.sigma, 1)).view(1, -1, 1)
        inv_gauss_k = inv_gauss_k.to(device).type(proj_range.type())

        k2_distances = k2_distances * inv_gauss_k

        _, knn_idx = k2_distances.topk(
            self.knn, dim=1, largest=False, sorted=False)

        proj_unfold_1_argmax = F.unfold(proj_argmax[None, None, ...].float(),
                                        kernel_size=(self.search, self.search),
                                        padding=(pad, pad)).long()
        unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list]

        knn_argmax = torch.gather(
            input=unproj_unfold_1_argmax, dim=1, index=knn_idx)

        if self.cutoff > 0:
            knn_distances = torch.gather(input=k2_distances, dim=1, index=knn_idx)
            knn_invalid_idx = knn_distances > self.cutoff
            knn_argmax[knn_invalid_idx] = self.nclasses

        knn_argmax_onehot = torch.zeros(
            (1, self.nclasses + 1, P[0]), device=device).type(proj_range.type())
        ones = torch.ones_like(knn_argmax).type(proj_range.type())
        knn_argmax_onehot = knn_argmax_onehot.scatter_add_(1, knn_argmax, ones)

        knn_argmax_out = knn_argmax_onehot[:, 1:-1].argmax(dim=1) + 1

        knn_argmax_out = knn_argmax_out.view(P)

        return knn_argmax_out
