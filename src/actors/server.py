import torch
import copy

from utils.model_computations import model_squared_norm, model_summation, model_subtraction, model_average, model_scale, \
    model_grad_cosine_similarity
from utils.general import setup_optimizer_and_scheduler


class Server:
    """
    Class for the server holding the global model.
    """
    def __init__(self, args, model, clients):
        self.args = args
        self.model = model
        self.clients = clients
        self.num_clients = len(clients)

        # FedDyn
        self.h = None

        # FedMix
        self.global_batch_mean_bank = None

        # FedDG
        self.global_amplitude_bank = None

        self.optimizer, _ = setup_optimizer_and_scheduler(
            args=self.args,
            model=self.model,
            scope="global"
        )

    def copy_model_to_clients(self):
        server_state_dict = self.model.state_dict()
        for client in self.clients:
            if client.model is None:
                # client models not initialized yet
                client.model = copy.deepcopy(self.model)
                # associate new optimizer with the initialized model
                client.create_new_optimizer()
            else:
                # update client models with server state dict
                client.model.load_state_dict(server_state_dict)

    def copy_representation_to_clients(self):
        server_state_dict = self.model.state_dict()

        for client in self.clients:
            client_state_dict = client.model.state_dict()
            for key in client_state_dict.keys():
                # keep head of client
                if key in client.representation_keys:
                    # copy server representation
                    client_state_dict[key] = server_state_dict[key]
            client.model.load_state_dict(client_state_dict)

    def aggregation(self, local_weights, active_clients):
        local_weight_diffs = []

        # compute "gradients" manually
        for i in range(len(local_weights)):  # active clients
            model_diff = model_subtraction(self.model.state_dict(), local_weights[i])
            local_weight_diffs.append(model_diff)

        if self.args.weighted_aggregation:
            num_total_samples = sum(len(self.clients[i].trainloader) for i in active_clients)
            data_share = [len(self.clients[i].trainloader) / num_total_samples for i in active_clients]
            global_weight_diff = model_average(local_weight_diffs, weights=data_share)
        else:
            global_weight_diff = model_average(local_weight_diffs)

        global_local_grad_lengths = []
        local_grad_lengths = []
        global_local_grad_cos_sims = []
        local_local_grad_cos_sims = []
        for client_idx in range(self.num_clients):  # all clients
            global_local_grad_lengths.append(model_squared_norm(model_subtraction(global_weight_diff, local_weight_diffs[client_idx])))
            local_grad_lengths.append(model_squared_norm(local_weight_diffs[client_idx]))
            global_local_grad_cos_sims.append(model_grad_cosine_similarity(global_weight_diff, local_weight_diffs[client_idx]))
            local_local_grad_cos_sims.append(
                [model_grad_cosine_similarity(local_weight_diffs[client_idx], local_weight_diffs[i]) for i in range(self.num_clients)]
            )
        grad_info = (global_local_grad_lengths, local_grad_lengths, global_local_grad_cos_sims, local_local_grad_cos_sims)

        # load "gradients" into optimizer and update weights
        self.optimizer.zero_grad()
        for key, param in self.model.named_parameters():
            param.grad = global_weight_diff[key]
        self.optimizer.step()
        return grad_info

    def fed_dyn_aggregation(self, local_weights, active_clients, alpha):
        local_weight_diffs = []

        # compute "gradients" manually
        for i in range(len(local_weights)):
            model_diff = model_subtraction(local_weights[i], self.model.state_dict())
            local_weight_diffs.append(model_diff)

        if self.args.weighted_aggregation:
            num_total_samples = sum(len(self.clients[i].trainloader) for i in active_clients)
            data_share = [len(self.clients[i].trainloader) / num_total_samples for i in active_clients]
            global_weight_diff = model_average(local_weight_diffs, weights=data_share)
        else:
            global_weight_diff = model_average(local_weight_diffs)

        # own implementation
        if self.h is None:
            self.h = model_scale(
                model=global_weight_diff,
                alpha=alpha  # -(alpha / self.num_clients)
            )
        else:
            self.h = model_subtraction(
                self.h,
                model_scale(
                    model=global_weight_diff,
                    alpha=alpha  # (alpha / self.num_clients)
                )
            )
        new_parameters = model_average([self.clients[i].model.state_dict() for i in active_clients])
        new_parameters = model_subtraction(new_parameters, model_scale(self.h, (1 / alpha)))

        global_local_grad_lengths = []
        local_grad_lengths = []
        global_local_grad_cos_sims = []
        local_local_grad_cos_sims = []
        for client_idx in range(self.num_clients):  # all clients
            global_local_grad_lengths.append(model_squared_norm(model_subtraction(global_weight_diff, local_weight_diffs[client_idx])))
            local_grad_lengths.append(model_squared_norm(local_weight_diffs[client_idx]))
            global_local_grad_cos_sims.append(model_grad_cosine_similarity(global_weight_diff, local_weight_diffs[client_idx]))
            local_local_grad_cos_sims.append(
                [model_grad_cosine_similarity(local_weight_diffs[client_idx], local_weight_diffs[i]) for i in range(self.num_clients)]
            )
        grad_info = (global_local_grad_lengths, local_grad_lengths, global_local_grad_cos_sims, local_local_grad_cos_sims)

        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = new_parameters[key]
        self.model.load_state_dict(state_dict)
        return grad_info

    def fed_rep_aggregation(self, local_weights, active_clients, representation_keys):
        local_weight_diffs = []

        # compute "gradients" manually
        for i in range(len(local_weights)):
            model_diff = model_subtraction(self.model.state_dict(), local_weights[i])
            local_weight_diffs.append(model_diff)

        if self.args.weighted_aggregation:
            num_total_samples = sum(len(self.clients[i].trainloader) for i in active_clients)
            data_share = [len(self.clients[i].trainloader) / num_total_samples for i in active_clients]
            global_weight_diff = model_average(local_weight_diffs, weights=data_share)
        else:
            global_weight_diff = model_average(local_weight_diffs)

        global_local_grad_lengths = []
        local_grad_lengths = []
        global_local_grad_cos_sims = []
        local_local_grad_cos_sims = []
        for client_idx in range(self.num_clients):  # all clients
            # only compute gradient on representation keys (without head)
            global_local_grad_lengths.append(model_squared_norm(model_subtraction(
                self.representation_of_model(global_weight_diff, representation_keys),
                self.representation_of_model(local_weight_diffs[client_idx], representation_keys))))
            local_grad_lengths.append(model_squared_norm(
                self.representation_of_model(local_weight_diffs[client_idx], representation_keys)))
            global_local_grad_cos_sims.append(model_grad_cosine_similarity(
                self.representation_of_model(global_weight_diff, representation_keys),
                self.representation_of_model(local_weight_diffs[client_idx], representation_keys)))
            local_local_grad_cos_sims.append(
                [model_grad_cosine_similarity(
                    self.representation_of_model(local_weight_diffs[client_idx], representation_keys),
                    self.representation_of_model(local_weight_diffs[i], representation_keys)) for i in range(self.num_clients)]
            )
        grad_info = (global_local_grad_lengths, local_grad_lengths, global_local_grad_cos_sims, local_local_grad_cos_sims)

        # load "gradients" into optimizer and update weights
        self.optimizer.zero_grad()
        for key, param in self.model.named_parameters():
            if key in representation_keys:
                param.grad = global_weight_diff[key]
            else:
                param.grad = torch.zeros_like(global_weight_diff[key])
        self.optimizer.step()
        return grad_info

    def accumulate_batch_mean_bank(self):
        bank = []
        for client in self.clients:
            bank.extend(client.local_batch_mean_bank)
        self.global_batch_mean_bank = bank

        for client in self.clients:
            client.global_batch_mean_bank = bank

    def accumulate_amplitude_bank(self):
        bank = {}
        for client in self.clients:
            bank[client.client_idx] = client.amplitude_bank
        self.global_amplitude_bank = bank

    def representation_of_model(self, model, representation_keys):
        res = copy.deepcopy(model)
        for key in model.keys():
            if key not in representation_keys:
                res[key] = torch.zeros_like(res[key])
        return res
