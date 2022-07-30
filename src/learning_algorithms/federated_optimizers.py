import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import copy
import glob
import matplotlib.pyplot as plt
from math import ceil
import time

from utils.save_results import clear_directory
from utils.general import setup_model, test_inference, setup_client_dataloaders, setup_data_split, send_message
from utils.save_results import ResultWriter
from actors.server import Server
from actors.client import ClientFedAvg, ClientFedProx, ClientFedDyn, ClientFedMix, ClientFedRep, ClientFedDG


def main(args, connection):
    send_message(connection, "-"*100)
    send_message(connection, f"Started Training for {args.mode}")

    # init env
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # define GPU is available, else use CPU
    print("Device:", device)
    torch.multiprocessing.set_sharing_strategy("file_system")

    # setup dataloaders
    data_split = setup_data_split(args)
    trainloaders, testloaders = setup_client_dataloaders(args, data_split)
    num_clients = len(trainloaders)
    print(trainloaders[0].dataset[0][2])
    num_classes = trainloaders[0].dataset[0][2].shape[0]

    # init model on cpu
    server_model = setup_model(args.model_name, num_classes)
    server_model.to(torch.device("cpu"))

    # logging to tensorboard and setup result tracking
    experiment_name = f"{args.mode}_{args.model_name}_{args.global_optimizer}{args.global_lr}"
    if num_classes == 6:
        experiment_name = f"{experiment_name}_6classes"
    if not args.weighted_aggregation:
        experiment_name = f"{experiment_name}_noweightedaggr"

    results_path = f"results/{experiment_name}"
    clear_directory(results_path)

    tensorboard_path = f"{results_path}/tensorboard"
    clear_directory(tensorboard_path)
    logger = SummaryWriter(tensorboard_path)

    labels = ["Grasper", "Clipper", "Coagulation", "Scissors", "Irrigation", "Specimen", "Stapler"][:num_classes]
    train_result_writer = ResultWriter(
        args=args,
        results_path=results_path,
        writer_type="train",
        labels=labels,
        num_clients=num_clients
    )

    if args.mode == "FedRep":
        test_result_writers = []

        for client_idx in range(num_clients):
            # subfolders for logging
            experiment_sub_name = f"client{str(client_idx)}"
            results_sub_path = f"{results_path}/{experiment_sub_name}"
            clear_directory(results_sub_path)

            test_result_writers.append(ResultWriter(
                args=args,
                results_path=results_sub_path,
                writer_type="test",
                labels=labels,
                num_clients=num_clients
            ))
    else:
        test_result_writer = ResultWriter(
            args=args,
            results_path=results_path,
            writer_type="test",
            labels=labels,
            num_clients=num_clients
        )

    # pw = inf is set to 1., since the label is not present then
    if args.pos_weight:
        pw = [
            torch.Tensor([0.60654685, 20.78890098, 1.77918112, 20.21186441, 8.08529946, 7.38174969, 1.]).to(device),
            torch.Tensor([0.49568148, 32.70253165, 0.71691117, 70.71717172, 18.77715877, 10.16352201, 1.]).to(device),
            torch.Tensor([0.97459337, 57.54392523, 0.71565513, 30.28971029, 14.18962173, 12.98258929, 1.]).to(device),
            torch.Tensor([0.88525193, 50.22341185, 0.68203445, 29.85296647, 18.29139785, 19.75303644, 1.]).to(device),
            torch.Tensor([0.68054592, 69.26112186,  0.55754223, 33.5952381, 18.8932092,  13.57664526, 1.]).to(device)
        ]
    else:
        pw = [None] * num_clients

    if args.mode == "FedAvg":
        clients = [ClientFedAvg(args=args, client_idx=i, trainloader=trainloaders[i], pos_weight=pw[i][:num_classes],
                                logger=logger) for i in range(num_clients)]
    elif args.mode == "FedProx":
        clients = [ClientFedProx(args=args, client_idx=i, trainloader=trainloaders[i], pos_weight=pw[i][:num_classes],
                                 logger=logger) for i in range(num_clients)]
    elif args.mode == "FedDyn":
        clients = [ClientFedDyn(args=args, client_idx=i, trainloader=trainloaders[i], pos_weight=pw[i][:num_classes],
                                logger=logger) for i in range(num_clients)]
    elif args.mode == "FedMix":
        clients = [ClientFedMix(args=args, client_idx=i, trainloader=trainloaders[i], pos_weight=pw[i][:num_classes],
                                logger=logger) for i in range(num_clients)]
    elif args.mode == "FedDG":
        clients = [ClientFedDG(args=args, client_idx=i, trainloader=trainloaders[i], pos_weight=pw[i][:num_classes],
                               logger=logger) for i in range(num_clients)]
    elif args.mode == "FedRep":
        # double the client epochs to split head and representation train epochs
        args.n_local_epochs *= 2
        # head epochs must be fraction of all epochs
        assert 0 < args.n_local_head_epochs < args.n_local_epochs
        clients = [ClientFedRep(args=args, client_idx=i, trainloader=trainloaders[i], pos_weight=pw[i][:num_classes],
                                logger=logger) for i in range(num_clients)]
    server = Server(args, server_model, clients)

    for key, p in server_model.named_parameters():
        if "weight" in key:
            assert torch.max(p) != 0

    local_weighted_results = []
    local_individual_results = []

    for epoch in tqdm(range(args.n_global_epochs), desc="Epoch"):
        print(f"\nCommunication Round: {epoch}\n")
        start_time = time.time()

        # train
        # sample active clients
        k = max(int(args.frac * num_clients), 1)
        active_clients = np.sort(np.random.choice(range(num_clients), k, replace=False))

        local_weights = []
        local_losses = []

        # compute local updates
        if args.mode == "FedAvg":
            server.copy_model_to_clients()
            for i in active_clients:
                client = server.clients[i]
                weights, loss, local_weighted_result, local_individual_result = client.federated_avg_update()
                torch.cuda.empty_cache()
                local_weights.append(weights)
                local_losses.append(copy.deepcopy(loss))
                local_weighted_results.append(local_weighted_result)
                local_individual_results.append(local_individual_result)

        elif args.mode == "FedProx":
            if epoch == 0:
                for i in range(num_clients):
                    server.clients[i].set_server_model(server_model)
            server.copy_model_to_clients()
            server_model.to(device)  # server model required during client training on device
            for i in active_clients:
                client = server.clients[i]
                weights, loss, local_weighted_result, local_individual_result = client.federated_prox_update()
                local_weights.append(weights)
                local_losses.append(copy.deepcopy(loss))
                local_weighted_results.append(local_weighted_result)
                local_individual_results.append(local_individual_result)
            server_model.to(torch.device("cpu"))

        elif args.mode == "FedDyn":
            server_model.to(device)  # server model required during client training on device
            if epoch == 0:
                server.copy_model_to_clients()
                for i in range(num_clients):
                    server.clients[i].set_server_model(server_model)
                    server.clients[i].initialize_local_grad()

            # alphas can be computed adaptively
            average_weights = np.array([len(trainloaders[i].dataset) for i in range(num_clients)])
            average_weights = average_weights / np.sum(average_weights) * num_clients
            alphas = args.alpha / average_weights
            print(alphas)

            for i in active_clients:
                client = server.clients[i]
                client.set_alpha(alphas[i])
                weights, loss, local_weighted_result, local_individual_result = client.federated_dyn_update()
                client.compute_local_gradient()
                local_weights.append(weights)
                local_losses.append(copy.deepcopy(loss))
                local_weighted_results.append(local_weighted_result)
                local_individual_results.append(local_individual_result)
            server_model.to(torch.device("cpu"))

        elif args.mode == "FedDG":
            server.copy_model_to_clients()
            for i in active_clients:
                client = server.clients[i]
                weights, loss, local_weighted_result, local_individual_result = client.federated_dg_update()
                local_weights.append(weights)
                local_losses.append(copy.deepcopy(loss))
                local_weighted_results.append(local_weighted_result)
                local_individual_results.append(local_individual_result)

        elif args.mode == "FedRep":
            if args.model_name == "alexnet":
                representation_keys = [
                    'features.0.weight', 'features.0.bias', 'features.3.weight', 'features.3.bias', 'features.6.weight',
                    'features.6.bias', 'features.8.weight', 'features.8.bias', 'features.10.weight', 'features.10.bias'
                ]
            elif args.model_name == "vgg13":
                print(server_model)
                representation_keys = [
                    'features.0.weight', 'features.0.bias', 
                    'features.2.weight', 'features.2.bias', 
                    'features.5.weight', 'features.5.bias', 
                    'features.7.weight', 'features.7.bias', 
                    'features.10.weight', 'features.10.bias', 
                    'features.12.weight', 'features.12.bias', 
                    'features.15.weight', 'features.15.bias', 
                    'features.17.weight', 'features.17.bias', 
                    'features.20.weight', 'features.20.bias', 
                    'features.22.weight', 'features.22.bias',
                ]
            if epoch == 0:
                server.copy_model_to_clients()
                for i in range(num_clients):
                    client = server.clients[i]
                    client.set_representation_keys(representation_keys)
            else:
                server.copy_representation_to_clients()

            for i in active_clients:
                client = server.clients[i]
                weights, loss, local_weighted_result, local_individual_result = client.federated_rep_update()
                local_weights.append(weights)
                local_losses.append(copy.deepcopy(loss))
                local_weighted_results.append(local_weighted_result)
                local_individual_results.append(local_individual_result)

        torch.cuda.memory_summary(device=None, abbreviated=False)

        # server update
        if args.mode == "FedDyn":
            grad_info = server.fed_dyn_aggregation(local_weights, active_clients, args.alpha)
        elif args.mode == "FedRep":
            grad_info = server.fed_rep_aggregation(local_weights, active_clients, representation_keys)
        else:
            grad_info = server.aggregation(local_weights, active_clients)
        logger.add_scalars("GradLengths", {'avg-global-local': np.mean([grad_info[0][i] for i in range(num_clients)])}, global_step=epoch)
        logger.add_scalars("GradLengths", {'avg-local': np.mean([grad_info[1][i] for i in range(num_clients)])}, global_step=epoch)
        logger.add_scalars("GradCosinusSimilaritiesGL", {'avg-global-local': np.mean([grad_info[2][i] for i in range(num_clients)])}, global_step=epoch)

        # logging train results
        global_loss = np.mean(local_losses)
        print("global loss", global_loss)

        logger.add_scalars("LossServer", {'train': global_loss}, global_step=epoch)
        logger.add_scalar("LR", server.clients[0].lr_scheduler.get_last_lr()[0], global_step=epoch)

        train_result_writer.write_loss_and_grads(global_loss, grad_info)
        train_result_writer.write_weighted_f1score("local", local_weighted_results)
        train_result_writer.write_individual_f1score("local", local_individual_results)

        # test
        if args.mode == "FedRep":
            results_list = []
            for client in server.clients:
                idx = client.client_idx
                global_weighted_results, global_macro_results, global_individual_results, \
                local_weighted_results, local_macro_results, local_individual_results =\
                    test_inference(copy.deepcopy(client.model), testloaders, torch.nn.BCEWithLogitsLoss())
                test_result_writers[idx].write_all_f1scores(
                    global_weighted_results, global_macro_results, global_individual_results,
                    local_weighted_results, local_macro_results, local_individual_results
                )
                results_list.append(global_macro_results)

                # save models
                test_result_writers[idx].save_best_model_if_improved(client.model, global_macro_results[2])
                criteria_met = test_result_writers[idx].save_model_if_stopping_criteria_met(client.model,
                                                                                            global_macro_results[2])

                # increment epoch counter in writers
                test_result_writers[idx].increment_epoch()
            train_result_writer.increment_epoch()

            results_list = np.asarray(results_list)
            num_total_samples = sum(len(trainloaders[i]) for i in range(num_clients))
            data_share = [len(trainloaders[i]) / num_total_samples for i in range(num_clients)]
            print(results_list.shape)
            mean_results = np.average(results_list, axis=0, weights=data_share)
            print(mean_results.shape)
            logger.add_scalars("LossServer", {'test': mean_results[3]}, global_step=epoch)

            logger.add_scalars("PrecisionTest", {'global': mean_results[0]}, global_step=epoch)
            logger.add_scalars("RecallTest", {'global': mean_results[1]}, global_step=epoch)
            logger.add_scalars("F1-ScoreTest", {'global': mean_results[2]}, global_step=epoch)
            f1score = mean_results[2]

        else:
            global_weighted_results, global_macro_results, global_individual_results, \
            local_weighted_results, local_macro_results, local_individual_results =\
                test_inference(
                    copy.deepcopy(server_model),
                    testloaders,
                    torch.nn.BCEWithLogitsLoss()
                )
            test_result_writer.write_all_f1scores(
                global_weighted_results, global_macro_results, global_individual_results,
                local_weighted_results, local_macro_results, local_individual_results
            )

            logger.add_scalars("LossServer", {'test': global_macro_results[3]}, global_step=epoch)

            logger.add_scalars("PrecisionTest", {'global': global_macro_results[0]}, global_step=epoch)
            logger.add_scalars("RecallTest", {'global': global_macro_results[1]}, global_step=epoch)
            logger.add_scalars("F1-ScoreTest", {'global': global_macro_results[2]}, global_step=epoch)
            f1score = global_macro_results[2]

            for client in server.clients:
                idx = client.client_idx
                logger.add_scalars("GradLengths", {f'global-local/client{idx}': grad_info[0][idx]}, global_step=epoch)
                logger.add_scalars("GradLengths", {f'local/client{idx}': grad_info[1][idx]}, global_step=epoch)
                logger.add_scalars("GradCosinusSimilaritiesGL", {f'global-local/client{idx}': grad_info[2][idx]}, global_step=epoch)
                logger.add_scalars("GradCosinusSimilaritiesLL",
                                   {f'avg-local-local/client{idx}': np.mean(grad_info[3][idx])-(1/num_clients)}, global_step=epoch)  # -(1/num_clients) because cos sim to itself (=1) included

                for i in range(idx+1, num_clients):
                    logger.add_scalars("GradCosinusSimilaritiesLL", {f'local-local/client{idx}-to-client{i}': grad_info[3][idx][i]}, global_step=epoch)

                logger.add_scalars("PrecisionTest", {f'client{idx}': local_macro_results[idx][0]},
                                   global_step=epoch)
                logger.add_scalars("RecallTest", {f'client{idx}': local_macro_results[idx][1]},
                                   global_step=epoch)
                logger.add_scalars("F1-ScoreTest", {f'client{idx}': local_macro_results[idx][2]},
                                   global_step=epoch)

            # save models
            test_result_writer.save_best_model_if_improved(server_model, global_macro_results[2])
            criteria_met = test_result_writer.save_model_if_stopping_criteria_met(server_model, global_macro_results[2])
            # if criteria_met:
            #     print(f"Loss is stagnant at global epoch {epoch}. Stopping training.")
            #     break

            # increment epoch counter in writers
            train_result_writer.increment_epoch()
            test_result_writer.increment_epoch()

        send_message(connection, f"{args.mode}/{args.model_name}: Finished epoch {epoch} with F1Score {f1score}")

        print("Epoch Time: ", time.time() - start_time)

    if args.mode == "FedRep":
        # revert epoch change
        args.n_local_epochs //= 2
        
        # save the final model
        for client in server.clients:
                idx = client.client_idx
                test_result_writers[idx].save_final_model(client.model)
    else:
        # save the final model
        test_result_writer.save_final_model(server_model)

    # clean up
    logger.close()

