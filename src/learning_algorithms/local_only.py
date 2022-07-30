import torch
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import copy

from utils.save_results import clear_directory, ResultWriter
from utils.general import setup_model, compute_pos_weight, test_inference, setup_data_split, setup_client_dataloaders, setup_optimizer_and_scheduler


def main(args):
    # init env
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # define GPU is available, else use CPU
    print("Device:", device)

    data_split = setup_data_split(args)
    trainloaders, testloaders = setup_client_dataloaders(args, data_split)
    num_clients = len(trainloaders)
    num_classes = trainloaders[0].dataset[0][2].shape[0]

    # logging to tensorboard and setup result tracking
    experiment_name = f"{args.mode}_{args.model_name}_{args.global_optimizer}{args.global_lr}"
    if num_classes == 6:
        experiment_name = f"{args.mode}_{args.model_name}_{args.global_optimizer}{args.global_lr}_6classes"

    results_path = f"results/{experiment_name}"
    clear_directory(results_path)

    sigmoid = torch.nn.Sigmoid()

    # pw = inf is set to 1., since the label is not present then
    if args.pos_weight:
        pw = [
            # torch.Tensor([0.5342557, 16.92307692, 1.71284668, 37.18968692, 8.1392684, 5.93777183, 1.]).to(device),
            # torch.Tensor([0.4849479, 32.75, 0.72967991, 69.46703297, 16.89534884, 11.70221195, 376.20588235]).to(device),
            torch.Tensor([0.60654685, 20.78890098, 1.77918112, 20.21186441, 8.08529946, 7.38174969, 1.]).to(device),
            torch.Tensor([0.49568148, 32.70253165, 0.71691117, 70.71717172, 18.77715877, 10.16352201, 1.]).to(device),
            torch.Tensor([0.97459337, 57.54392523, 0.71565513, 30.28971029, 14.18962173, 12.98258929, 1.]).to(device),
            torch.Tensor([0.88525193, 50.22341185, 0.68203445, 29.85296647, 18.29139785, 19.75303644, 1.]).to(device),
            torch.Tensor([0.68054592, 69.26112186,  0.55754223, 33.5952381, 18.8932092,  13.57664526, 1.]).to(device)
        ]
    else:
        pw = [None] * num_clients

    for client_idx in range(num_clients):  # range(2, num_clients):  # range(num_clients):
        # subfolders for logging
        experiment_sub_name = f"client{str(client_idx)}"
        results_sub_path = f"{results_path}/{experiment_sub_name}"
        clear_directory(results_sub_path)

        tensorboard_path = f"{results_sub_path}/tensorboard"
        clear_directory(tensorboard_path)
        logger = SummaryWriter(tensorboard_path)

        labels = ["Grasper", "Clipper", "Coagulation", "Scissors", "Irrigation", "Specimen", "Stapler"][:num_classes]
        train_result_writer = ResultWriter(
            args=args,
            results_path=results_sub_path,
            writer_type="train",
            labels=labels,
            num_clients=num_clients
        )
        test_result_writer = ResultWriter(
            args=args,
            results_path=results_sub_path,
            writer_type="test",
            labels=labels,
            num_clients=num_clients
        )

        trainloader = trainloaders[client_idx]
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw[client_idx][num_classes])

        # init model
        model = setup_model(args.model_name, num_classes)
        model = model.to(device)

        optimizer, lr_scheduler = setup_optimizer_and_scheduler(
            args=args,
            model=model,
            scope="global",
            len_trainloader=len(trainloader)
        )

        for epoch in tqdm(range(args.n_global_epochs), desc="Epoch"):

            # train
            model.train()
            losses = []
            predictions = []
            targets = []

            for _, input, target in tqdm(trainloader, desc="Train", leave=False):
                input = input.to(device)

                # forward pass
                output = model(input)
                predictions.extend((sigmoid(output) >= 0.5).detach().cpu().numpy())
                targets.extend(target.numpy())

                loss = criterion(output, target.to(device).float())
                losses.append(loss.item())

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            # logging train results
            logger.add_scalar("LR", lr_scheduler.get_last_lr()[0], global_step=epoch)
            logger.add_scalars("Loss", {'train': np.mean(losses)}, global_step=epoch)
            train_result_writer.write_loss(np.mean(losses))

            precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted',
                                                                       zero_division=0)
            logger.add_scalars("Precision", {'train': precision}, global_step=epoch)
            logger.add_scalars("Recall", {'train': recall}, global_step=epoch)
            logger.add_scalars("F1-Score", {'train': f1}, global_step=epoch)
            train_result_writer.write_weighted_f1score("global", [precision, recall, f1, np.mean(losses)])

            precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average=None,
                                                                       zero_division=0)
            train_result_writer.write_individual_f1score("global", [precision, recall, f1, np.mean(losses)])

            # test
            global_weighted_results, global_macro_results, global_individual_results, \
            local_weighted_results, local_macro_results, local_individual_results =\
                test_inference(copy.deepcopy(model), testloaders, criterion)
            logger.add_scalars("Loss", {'test': local_macro_results[client_idx][3]}, global_step=epoch)

            # logging test results
            for idx in range(num_clients):
                logger.add_scalars("Precision", {f'test/client{idx}': local_macro_results[idx][0]}, global_step=epoch)
                logger.add_scalars("Recall", {f'test/client{idx}': local_macro_results[idx][1]}, global_step=epoch)
                logger.add_scalars("F1-Score", {f'test/client{idx}': local_macro_results[idx][2]}, global_step=epoch)

            test_result_writer.write_all_f1scores(
                global_weighted_results, global_macro_results, global_individual_results,
                local_weighted_results, local_macro_results, local_individual_results
            )
            test_result_writer.save_best_model_if_improved(model, local_macro_results[client_idx][2])
            criteria_met = test_result_writer.save_model_if_stopping_criteria_met(model,
                                                                                  local_macro_results[client_idx][2])
            # if criteria_met:
            #     print(f"Loss is stagnant at global epoch {epoch}. Stopping training.")
            #     break

            # increment epoch counter in writers
            train_result_writer.increment_epoch()
            test_result_writer.increment_epoch()

        # save the final model
        test_result_writer.save_final_model(model)

        # clean up
        logger.close()
        model.to(torch.device("cpu"))
