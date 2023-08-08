import logging
import math
import time
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sknetwork.ranking import PageRank
from torch.optim import lr_scheduler

from diagnoser.daggnn.config import Config
from diagnoser.daggnn.modules import (MLPDecoder, MLPEncoder, SEMDecoder,
                                      SEMEncoder)
from diagnoser.daggnn.utils import (A_connect_loss, A_positive_loss, Variable,
                                    get_tril_offdiag_indices,
                                    get_triu_offdiag_indices, kl_gaussian_sem,
                                    matrix_poly, nll_gaussian)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
st_handler = logging.StreamHandler()
logger.addHandler(st_handler)


class CausalRCA:
    def __init__(self, data: pd.DataFrame, conf: Config, num_threads: int = 1):
        self.conf = conf

        conf.cuda = torch.cuda.is_available()
        torch.manual_seed(conf.seed)
        if conf.cuda:
            torch.cuda.manual_seed(conf.seed)
        torch.set_num_threads(num_threads)

        self.data = data
        self.data_sample_size = data.shape[0]
        self.data_variable_size = data.shape[1]
        self.train_data = data

        batch_size: int = self.data_sample_size * self.conf.sample_to_batch_size_factor
        # off_diag = np.ones([self.data_variable_size, self.data_variable_size]) - np.eye(self.data_variable_size)

        # add adjacency matrix A
        adj_A = np.zeros((self.data_variable_size, self.data_variable_size))

        match conf.encoder:
            case "mlp":
                self.encoder = MLPEncoder(
                    self.data_variable_size * conf.x_dims,
                    conf.x_dims,
                    conf.encoder_hidden,
                    conf.z_dims,
                    adj_A,
                    batch_size=batch_size,
                    do_prob=conf.encoder_dropout,
                    factor=conf.factor,
                ).double()
            case "sem":
                self.encoder = SEMEncoder(
                    self.data_variable_size * conf.x_dims,
                    conf.encoder_hidden,
                    conf.z_dims,
                    adj_A,
                    batch_size=batch_size,
                    do_prob=conf.encoder_dropout,
                    factor=conf.factor,
                ).double()
            case _:
                raise ValueError(f"{conf.encoder} is not a valid encoder")

        match conf.decoder:
            case "mlp":
                self.decoder = MLPDecoder(
                    self.data_variable_size * conf.x_dims,
                    conf.z_dims,
                    conf.x_dims,
                    self.encoder,
                    data_variable_size=self.data_variable_size,
                    batch_size=batch_size,
                    n_hid=conf.decoder_hidden,
                    do_prob=conf.decoder_dropout,
                ).double()
            case "sem":
                self.decoder = SEMDecoder(
                    self.data_variable_size * conf.x_dims,
                    conf.z_dims,
                    2,
                    self.encoder,
                    data_variable_size=self.data_variable_size,
                    batch_size=batch_size,
                    n_hid=conf.decoder_hidden,
                    do_prob=conf.decoder_dropout,
                ).double()
            case _:
                raise ValueError(f"{conf.decoder} is not a valid decoder")

        match conf.optimizer:
            case "Adam":
                self.optimizer = optim.Adam(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=conf.lr
                )
            case "LBFGS":
                self.optimizer = optim.LBFGS(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=conf.lr
                )
            case "SGD":
                self.optimizer = optim.SGD(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=conf.lr
                )
            case _:
                raise ValueError(f"{conf.optimizer} is not a valid optimizer")

        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=conf.lr_decay, gamma=conf.sche_gamma)
        # Linear indices of an upper triangular mx, used for acc calculation
        triu_indices = get_triu_offdiag_indices(self.data_variable_size)
        tril_indices = get_tril_offdiag_indices(self.data_variable_size)
        if conf.prior:
            prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
            log_prior = torch.DoubleTensor(np.log(prior))
            log_prior = torch.unsqueeze(log_prior, 0)
            log_prior = torch.unsqueeze(log_prior, 0)
            log_prior = Variable(log_prior)

            if conf.cuda:
                log_prior = log_prior.cuda()

        if conf.cuda:
            self.encoder.cuda()
            self.decoder.cuda()
            triu_indices = triu_indices.cuda()
            tril_indices = tril_indices.cuda()

        self.prox_plus = torch.nn.Threshold(0.0, 0.0)

    # compute constraint h(A) value
    @staticmethod
    def _h_A(A, m):
        expm_A = matrix_poly(A * A, m)
        h_A = torch.trace(expm_A) - m
        return h_A

    def stau(self, w, tau):
        w1 = self.prox_plus(torch.abs(w) - tau)
        return torch.sign(w) * w1

    def update_optimizer(self, original_lr, c_A):
        """related LR to c_A, whenever c_A gets big, reduce LR proportionally"""
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in self.optimizer.param_groups:
            parame_group["lr"] = lr

        return self.optimizer, lr

    def train(self, lambda_A, c_A):
        nll_train = []
        kl_train = []
        mse_train = []

        self.encoder.train()
        self.decoder.train()

        # update optimizer
        optimizer, lr = self.update_optimizer(self.conf.lr, c_A)

        for i in range(1):
            data = self.train_data[i * self.data_sample_size : (i + 1) * self.data_sample_size]
            data = torch.tensor(data.to_numpy().reshape(self.data_sample_size, self.data_variable_size, 1))
            if self.conf.cuda:
                data = data.cuda()
            data = Variable(data).double()

            optimizer.zero_grad()

            enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = self.encoder(
                data
            )  # logits is of size: [num_sims, z_dims]
            edges = logits
            # print(origin_A)
            dec_x, output, adj_A_tilt_decoder = self.decoder(
                data, edges, self.data_variable_size * self.conf.x_dims, origin_A, adj_A_tilt_encoder, Wa
            )

            if torch.sum(output != output):
                logger.info("nan error")

            target = data
            preds = output
            variance = 0.0

            # reconstruction accuracy loss
            loss_nll = nll_gaussian(preds, target, variance)

            # KL loss
            loss_kl = kl_gaussian_sem(logits)

            # ELBO loss:
            loss = loss_kl + loss_nll
            # add A loss
            one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = self.conf.tau_A * torch.sum(torch.abs(one_adj_A))

            # other loss term
            if self.conf.use_A_connect_loss:
                connect_gap = A_connect_loss(one_adj_A, self.conf.graph_threshold, z_gap)
                loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

            if self.conf.use_A_positiver_loss:
                positive_gap = A_positive_loss(one_adj_A, z_positive)
                loss += 0.1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

            # compute h(A)
            h_A = self._h_A(origin_A, self.data_variable_size)
            loss += (
                lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100.0 * torch.trace(origin_A * origin_A) + sparse_loss
            )  # +  0.01 * torch.sum(variance * variance)

            loss.backward()
            loss = optimizer.step()

            myA.data = self.stau(myA.data, self.conf.tau_A * lr)

            if torch.sum(origin_A != origin_A):
                logger.info("nan error")

            # compute metrics
            graph = origin_A.data.clone().cpu().numpy()
            graph[np.abs(graph) < self.conf.graph_threshold] = 0

            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())

        # scheduler should be stepped after optimizer.step()
        self.scheduler.step()

        return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A

    def fit(self) -> tuple[np.ndarray, float]:
        best_ELBO_loss, best_NLL_loss, best_MSE_loss = np.inf, np.inf, np.inf
        best_epoch = 0
        best_ELBO_graph, best_NLL_graph, best_MSE_graph = [], [], []
        # optimizer step on hyparameters
        c_A = self.conf.c_A
        lambda_A = self.conf.lambda_A
        h_A_new = torch.tensor(1.0)
        h_tol = self.conf.h_tol
        k_max_iter = int(self.conf.k_max_iter)
        h_A_old = np.inf

        E_loss, N_loss, M_loss = [], [], []

        start_time = time.time()
        try:
            for step_k in range(k_max_iter):
                while c_A < self.conf.c_A_ul:
                    ELBO_loss, NLL_loss, MSE_loss = 0.0, 0.0, 0.0
                    for epoch in range(self.conf.epochs):
                        ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = self.train(lambda_A, c_A)
                        E_loss.append(ELBO_loss)
                        N_loss.append(NLL_loss)
                        M_loss.append(MSE_loss)
                        if ELBO_loss < best_ELBO_loss:
                            best_ELBO_loss = ELBO_loss
                            best_epoch = epoch
                            best_ELBO_graph = graph
                        if NLL_loss < best_NLL_loss:
                            best_NLL_loss = NLL_loss
                            best_epoch = epoch
                            best_NLL_graph = graph
                        if MSE_loss < best_MSE_loss:
                            best_MSE_loss = MSE_loss
                            best_epoch = epoch
                            best_MSE_graph = graph

                    logging.info(
                        f"Step: {step_k+1}/{k_max_iter}, Best Epoch: {best_epoch+1}/{self.conf.epochs}, ELBO Loss: {best_ELBO_loss}, NLL Loss: {best_NLL_loss}, MSE Loss: {best_MSE_loss}"
                    )

                    if ELBO_loss > 2 * best_ELBO_loss:
                        break

                    # update parameters
                    A_new = origin_A.data.clone()
                    h_A_new = self._h_A(A_new, self.data_variable_size)
                    if h_A_new.item() > self.conf.gamma * h_A_old:
                        c_A *= self.conf.eta
                    else:
                        break

                # update parameters
                # h_A, adj_A are computed in loss anyway, so no need to store
                h_A_old = h_A_new.item()
                lambda_A += c_A * h_A_new.item()

                if h_A_new.item() <= h_tol:
                    break

            graph = origin_A.data.clone().cpu().numpy()
            # graph[np.abs(graph) < 0.1] = 0
            # graph[np.abs(graph) < 0.2] = 0
            # graph[np.abs(graph) < 0.3] = 0

        except KeyboardInterrupt:
            print("Done!")

        end_time = time.time()
        return graph, end_time - start_time

    def rank(self, graph_adj: np.ndarray, **params: Any) -> dict[str, tuple[str, float]]:
        # PageRank in networkx
        # G = nx.from_numpy_matrix(adj.T, parallel_edges=True, create_using=nx.DiGraph)
        # scores = nx.pagerank(G, max_iter=1000)
        # print(sorted(scores.items(), key=lambda item:item[1], reverse=True))
        pagerank = PageRank(solver=params.get("pagerank_solver", "piteration"))
        scores = pagerank.fit_transform(np.abs(graph_adj.T))
        score_dict = {}
        for i, s in enumerate(scores):
            score_dict[i] = s
        no_to_metric_name = {i: col for i, col in enumerate(self.data.columns)}
        sorted_dict = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        return {
            sorted_dict[i][0]: (no_to_metric_name[sorted_dict[i][0]], sorted_dict[i][1])
            for i in range(len(sorted_dict))
        }


def build_and_walk_causal_graph(data: pd.DataFrame, **kwargs: Any) -> tuple[nx.DiGraph, list[tuple[str, float]]]:
    conf = Config.from_prefixed_dict(**kwargs, prefix="causalrca")
    model = CausalRCA(data, conf)
    graph_adj, elapsed = model.fit()
    return nx.DiGraph(graph_adj), list(model.rank(graph_adj).values())
