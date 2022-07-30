import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from math import ceil
from typing import Callable
from functools import wraps
from utils.model_computations import model_subtraction, model_parameter_sum, \
    model_scalar_product, model_squared_norm, model_scale, \
    model_multiplication, model_summation
from utils.general import setup_optimizer_and_scheduler, compute_pos_weight
import copy
import torch.autograd as autograd
import gc


class Client:
    """
    Base class for the clients performing updates for the global objective.
    """

    class Decorators:
        @classmethod
        def loss_decorator(cls, loss_computation: Callable):
            # @wraps(loss_computation)
            def wrapper(self):  # , *args, **kwargs
                self.model.to(self.device)
                self.criterion.to(self.device)

                self.model.train()
                for param in self.model.parameters():
                    param.requires_grad = True

                for epoch in range(self.args.n_local_epochs):
                    # train
                    losses = []
                    predictions = []
                    targets = []

                    step = 0

                    for _, input, target in tqdm(self.trainloader, desc=f"Train Client {self.client_idx}", leave=False):
                        input = input.to(self.device)

                        # forward pass
                        output = self.model(input)
                        loss = loss_computation(self, output, target.to(self.device).float(), step)

                        predictions.extend((self.sigmoid(output) >= 0.5).detach().cpu().numpy())
                        targets.extend(target.numpy())

                        losses.append(loss.item())

                        # backward pass
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        step += 1

                    # its possible to track the local training epochs here
                    # precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions,
                    #                                                            average='weighted', zero_division=0)
                    # precisions.append(precision)
                    # recalls.append(recall)
                    # f1s.append(f1)

                for param_group in self.optimizer.param_groups:
                    print(param_group["lr"])

                precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted',
                                                                           zero_division=0)
                mean_loss = np.mean(losses)
                local_weighted_result = [precision, recall, f1, mean_loss]

                # logging train results
                self.logger.add_scalars("LossClients", {f'train/client{self.client_idx}': mean_loss},
                                        global_step=self.global_epoch)

                self.logger.add_scalars("PrecisionTrain", {f'client{self.client_idx}': precision},
                                        global_step=self.global_epoch)
                self.logger.add_scalars("RecallTrain", {f'client{self.client_idx}': recall},
                                        global_step=self.global_epoch)
                self.logger.add_scalars("F1-ScoreTrain", {f'client{self.client_idx}': f1},
                                        global_step=self.global_epoch)

                precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average=None,
                                                                           zero_division=0)
                local_individual_result = [precision, recall, f1, mean_loss]

                self.global_epoch += 1
                self.model.to(torch.device("cpu"))
                self.criterion.to(torch.device("cpu"))
                del input, output, loss
                return self.model.state_dict(), mean_loss, local_weighted_result, local_individual_result
            return wrapper

        @classmethod
        def dataloader_iterator_decorator(cls, dataloader_iteration: Callable):
            # @wraps(loss_computation)
            def wrapper(self):  # , *args, **kwargs
                self.model.to(self.device)
                self.model.train()
                for param in self.model.parameters():
                    param.requires_grad = True

                for epoch in range(self.args.n_local_epochs):
                    # train
                    losses, targets, predictions = dataloader_iteration(
                        self, epoch
                    )

                precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted',
                                                                           zero_division=0)
                mean_loss = np.mean(losses)
                local_weighted_result = [precision, recall, f1, mean_loss]

                # logging train results
                self.logger.add_scalars("LossClients", {f'train/client{self.client_idx}': mean_loss},
                                        global_step=self.global_epoch)

                self.logger.add_scalars("PrecisionTrain", {f'client{self.client_idx}': precision},
                                        global_step=self.global_epoch)
                self.logger.add_scalars("RecallTrain", {f'client{self.client_idx}': recall},
                                        global_step=self.global_epoch)
                self.logger.add_scalars("F1-ScoreTrain", {f'client{self.client_idx}': f1},
                                        global_step=self.global_epoch)

                precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average=None,
                                                                           zero_division=0)
                local_individual_result = [precision, recall, f1, mean_loss]

                self.global_epoch += 1
                self.model.to(torch.device("cpu"))
                return self.model.state_dict(), mean_loss, local_weighted_result, local_individual_result
            return wrapper

    def __init__(self, args, client_idx, trainloader, pos_weight=None, logger=None):
        self.args = args
        self.client_idx = client_idx
        self.trainloader = trainloader
        self.logger = logger

        self.global_epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Client {client_idx} on {self.device}")
        self.sigmoid = torch.nn.Sigmoid()

        print(compute_pos_weight(self.trainloader))

        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.server_model = None

    def __repr__(self):
        return "Client " + str(self.client_idx)

    def create_new_optimizer(self):
        self.optimizer, self.lr_scheduler = setup_optimizer_and_scheduler(
            args=self.args,
            model=self.model,
            scope="local",
            len_trainloader=len(self.trainloader)
        )

    def set_server_model(self, server_model):
        if server_model == self.server_model:
            print("same object")
        self.server_model = server_model


class ClientFedAvg(Client):
    def __init__(self, args, client_idx, trainloader, pos_weight=None, logger=None):
        super().__init__(args, client_idx, trainloader, pos_weight, logger)

    @Client.Decorators.loss_decorator
    def federated_avg_update(self, output, target, step=None):
        return self.criterion(output, target.to(self.device).float())


class ClientFedProx(Client):
    def __init__(self, args, client_idx, trainloader, pos_weight=None, logger=None):
        super().__init__(args, client_idx, trainloader, pos_weight, logger)

    @Client.Decorators.loss_decorator
    def federated_prox_update(self, output, target, step):
        loss = self.criterion(output, target.to(self.device).float())

        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        # FedProx

        if step > 0:
            w_diff = torch.tensor(0., device=self.device)
            for w, w_t in zip(self.model.parameters(), self.server_model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += self.args.mu / 2. * w_diff
        return loss

    # loss_algo = mu / 2 * torch.sum(local_par_list * local_par_list)
    # loss_algo = -mu * torch.sum(local_par_list * avg_model_param_)
    # loss = loss_f_i + loss_algo


class ClientFedDyn(Client):
    def __init__(self, args, client_idx, trainloader, pos_weight=None, logger=None):
        self.alpha = None
        self.local_grad = None  # grad
        self.constant_term = None
        super().__init__(args, client_idx, trainloader, pos_weight, logger)

    def initialize_local_grad(self):
        # initialize local_grad model to all 0
        local_grad = copy.deepcopy(self.server_model)
        state_dict = local_grad.state_dict()
        for key in state_dict.keys():
            state_dict[key] = torch.zeros_like(state_dict[key])
        self.local_grad = state_dict

    def set_alpha(self, alpha):
        self.alpha = alpha

    def compute_local_gradient(self):
        self.model.to(self.device)

        if self.args.own_implementation:
            # own implementation
            if self.local_grad is None:  # gets initially
                self.local_grad = model_scale(
                    model=model_subtraction(self.model.state_dict(), self.server_model.state_dict()),
                    alpha=-self.alpha
                )
            else:
                self.local_grad = model_subtraction(
                    self.local_grad,
                    model_scale(
                        model=model_subtraction(self.model.state_dict(), self.server_model.state_dict()),
                        alpha=self.alpha
                    )
                )

        else:
            # referring to https://github.com/alpemreacar/FedDyn/blob/master/utils_general.py
            # FedDyn
            self.local_grad = model_summation(
                self.local_grad,
                model_subtraction(
                    self.model.state_dict(),
                    self.server_model.state_dict()
                )
            )

        # for key in self.local_grad.keys():
        #     print(torch.max(self.local_grad[key]))
        self.model.to(torch.device("cpu"))

    def compute_constant_loss_term(self):
        self.constant_term = model_subtraction(
            self.local_grad,
            self.server_model.state_dict()
        )

    @Client.Decorators.loss_decorator
    def federated_dyn_update(self, output, target, step=None):
        loss = self.criterion(output, target.to(self.device).float())

        if self.args.own_implementation:
            # own implementation -> increasing train loss while improving performance
            if step > 0:
                # scalar_prod = model_scalar_product(self.local_grad, self.model.state_dict())
                # norm = (self.alpha / 2) * model_squared_norm(
                #     model_subtraction(self.model.state_dict(), self.server_model.state_dict())
                # )
                # # print("norm", norm)
                # # print("scal", scalar_prod)
                # loss += norm - scalar_prod
                loss += (self.alpha / 2) * model_squared_norm(
                    model_subtraction(self.model.state_dict(), self.server_model.state_dict())
                )
                loss -= model_scalar_product(self.local_grad, self.model.state_dict())

        else:
            # referring to https://github.com/alpemreacar/FedDyn/blob/master/utils_general.py
            # FedDyn -> highly negative loss
            loss += self.alpha * model_parameter_sum(
                model_multiplication(
                    self.model.state_dict(),
                    self.constant_term
                )
            )
        return loss


class ClientFedDG(Client):
    def __init__(self, args, client_idx, trainloader, pos_weight=None, logger=None):
        super().__init__(args, client_idx, trainloader, pos_weight, logger)

    # FedAvg training with additional augmentation module in trainloaders
    @Client.Decorators.loss_decorator
    def federated_dg_update(self, output, target, step=None):
        return self.criterion(output, target.to(self.device).float())


class ClientFedRep(Client):
    def __init__(self, args, client_idx, trainloader, pos_weight=None, logger=None):
        self.representation_keys = None
        super().__init__(args, client_idx, trainloader, pos_weight, logger)

    def set_representation_keys(self, representation_keys):
        self.representation_keys = representation_keys

    @Client.Decorators.dataloader_iterator_decorator
    def federated_rep_update(self, epoch):
        losses = []
        predictions = []
        targets = []

        # first do local update epochs for head
        if epoch < self.args.n_local_head_epochs:
            for key, param in self.model.named_parameters():
                if key in self.representation_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        # after n_local_head_epochs do representation updates
        else:
            for key, param in self.model.named_parameters():
                if key in self.representation_keys:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for _, input, target in tqdm(self.trainloader, desc=f"Train Client {self.client_idx}", leave=False):
            input = input.to(self.device)

            # forward pass
            output = self.model(input)
            loss = self.criterion(output, target.to(self.device).float())

            predictions.extend((self.sigmoid(output) >= 0.5).detach().cpu().numpy())
            targets.extend(target.numpy())
            losses.append(loss.item())

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch >= self.args.n_local_head_epochs:
                self.lr_scheduler.step()

        del input, output, loss
        return losses, targets, predictions
