import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import Cohort
from utils.save_results import clear_directory, ResultWriter
from utils.general import setup_model, test_inference, setup_data_split, setup_optimizer_and_scheduler, compute_pos_weight
import copy


def main(args):
    # init env
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # define GPU is available, else use CPU
    print("Device:", device)

    data_split = setup_data_split(args)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalization for images
    transform = transforms.Compose([transforms.ToTensor(), normalize])  # applies normalization and wraps into tensor

    # init data
    trainset = Cohort(
        args.data_root,  # path to the data
        data_split["train"][0],  # list of surgeries to be contained
        num_classes=args.n_classes,
        transform=transform,  # apply transformation on loading
        width=args.width,  # resize the images to desired size
        height=args.height,
        load_to_ram=args.load_to_ram,
    )
    print(trainset)
    trainloader = DataLoader(
        dataset=trainset,  # what data
        batch_size=args.local_batch_size,  # size of the chunks (batch) which are given to the model at once
        shuffle=True,  # random order while trainings
        num_workers=args.n_workers,  # workers load data in parallel
        pin_memory=args.pin_memory,  # speed up data loading
    )

    testloaders = {}
    test_data_split = data_split["test"]
    num_clients = len(test_data_split)
    num_classes = trainloader.dataset[0][2].shape[0]

    for client_idx, client_data in test_data_split.items():
        print(f"test {client_idx}")
        testset = Cohort(
            args.data_root,  # path to the data
            client_data,  # list of surgeries to be contained
            num_classes=args.n_classes,
            transform=transform,  # apply transformation on loading
            width=args.width,  # resize the images to desired size
            height=args.height,
            load_to_ram=args.load_to_ram,
        )
        print(testset)
        testloader = DataLoader(
            dataset=testset,  # what data
            batch_size=args.local_batch_size,  # size of the chunks (batch) which are given to the model at once
            shuffle=True,  # random order while testings
            num_workers=args.n_workers,  # workers load data in parallel
            pin_memory=args.pin_memory,  # speed up data loading
        )
        testloaders[client_idx] = testloader

    # logging to tensorboard and setup result tracking
    experiment_name = f"{args.mode}_{args.model_name}_{args.global_optimizer}{args.global_lr}"
    if num_classes == 6:
        experiment_name = f"{args.mode}_{args.model_name}_{args.global_optimizer}{args.global_lr}_6classes"

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
    test_result_writer = ResultWriter(
        args=args,
        results_path=results_path,
        writer_type="test",
        labels=labels,
        num_clients=num_clients
    )

    # init model
    model = setup_model(args.model_name, num_classes)
    model = model.to(device)
    print(model)

    optimizer, lr_scheduler = setup_optimizer_and_scheduler(
        args=args,
        model=model,
        scope="global",
        len_trainloader=len(trainloader)
    )

    sigmoid = torch.nn.Sigmoid()
    pw = None
    if args.pos_weight:
        #         pw = torch.Tensor([7.27613531e-01, 4.08151579e+01, 7.40298429e-01, 3.53377241e+01, 1.55146753e+01,
        #                            1.29795890e+01, 1.94627451e+03]).to(device)
        pw = torch.Tensor([0.76760426, 44.13836164,  0.74026999, 31.17052332, 15.59785104, 13.47377272]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw[:num_classes])

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

        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='macro',
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
        local_weighted_results, local_macro_results, local_individual_results = \
            test_inference(copy.deepcopy(model), testloaders, criterion)
        logger.add_scalars("Loss", {'test': global_macro_results[3]}, global_step=epoch)

        # logging test results
        logger.add_scalars("Precision", {'test': global_macro_results[0]}, global_step=epoch)
        logger.add_scalars("Recall", {'test': global_macro_results[1]}, global_step=epoch)
        logger.add_scalars("F1-Score", {'test': global_macro_results[2]}, global_step=epoch)

        test_result_writer.write_all_f1scores(
            global_weighted_results, global_macro_results, global_individual_results,
            local_weighted_results, local_macro_results, local_individual_results
        )
        test_result_writer.save_best_model_if_improved(model, global_macro_results[2])
        criteria_met = test_result_writer.save_model_if_stopping_criteria_met(model, global_macro_results[2])
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
