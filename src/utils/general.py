import torch
import torchvision
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import glob
import requests
import logging
import traceback

from utils.dataset import Cohort
from utils.augmentations import AmplitudeInterpolation


def setup_data_split(args):
    """
    args.data_split_type options: real, heichole, cholec-nct

    NOTE: client names need to be indices from 0 to num_clients
    """
    if args.data_split_type == "real":
        # real split
        if args.mode == "centralized":
            data_split = {
                "train": {
                    0: {"HeiChole": [2, 3, 4, 5, 6, 8, 9, 10, 12] +
                                    [13, 14, 16, 17, 19, 20, 21, 22, 24],
                        "Cholec80": [4, 12, 13, 24, 25, 37, 45, 46, 47, 48, 49, 53, 70, 72, 73, 76] +
                                    [1, 2, 5, 6, 9, 15, 16, 17, 29, 32, 36, 41, 42, 43, 50, 51, 58, 61, 62, 66, 69, 77,
                                     78,  39, 43, 52, 55, 56, 74, 79] +
                                    [3, 10, 18, 20, 21, 22, 31, 34, 38, 44, 60, 64, 71, 75, 80]}
                },
                "test": {
                    0: {"HeiChole": [1, 7, 11]},
                    1: {"HeiChole": [15, 18, 23]},
                    2: {"Cholec80": [7, 8, 19, 23, 26, 33, 67]},
                    3: {"Cholec80": [11, 27, 30, 54, 63, 68]},
                    4: {"Cholec80": [14, 28, 35, 40, 57, 59, 65]}
                }
            }
        else:  # federated training
            data_split = {
                "train": {
                    0: {"HeiChole": [2, 3, 4, 5, 6, 8, 9, 10, 12]},
                    1: {"HeiChole": [13, 14, 16, 17, 19, 20, 21, 22, 24]},
                    2: {"Cholec80": [4, 12, 13, 24, 25, 37, 45, 46, 47, 48, 49, 53, 70, 72, 73, 76]},
                    3: {"Cholec80": [1, 2, 5, 6, 9, 15, 16, 17, 29, 32, 36, 41, 42, 43, 50, 51, 58, 61, 62, 66, 69, 77, 78,
                                     39, 43, 52, 55, 56, 74, 79]},
                    4: {"Cholec80": [3, 10, 18, 20, 21, 22, 31, 34, 38, 44, 60, 64, 71, 75, 80]}
                },
                "test": {
                    0: {"HeiChole": [1, 7, 11]},
                    1: {"HeiChole": [15, 18, 23]},
                    2: {"Cholec80": [7, 8, 19, 23, 26, 33, 67]},
                    3: {"Cholec80": [11, 27, 30, 54, 63, 68]},
                    4: {"Cholec80": [14, 28, 35, 40, 57, 59, 65]}
                }
            }
    elif args.data_split_type == "heichole":
        # only heichole
        if args.mode == "centralized":
            data_split = {
                "train": {
                    0: {"HeiChole": [2, 3, 4, 5, 6, 8, 9, 10, 12] +
                                    [13, 14, 16, 17, 19, 20, 21, 22, 24]}
                },
                "test": {
                    0: {"HeiChole": [1, 7, 11]},
                    1: {"HeiChole": [15, 18, 23]}
                }
            }
        else:  # federated training
            data_split = {
                "train": {
                    0: {"HeiChole": [2, 3, 4, 5, 6, 8, 9, 10, 12]},
                    1: {"HeiChole": [13, 14, 16, 17, 19, 20, 21, 22, 24]},
                },
                "test": {
                    0: {"HeiChole": [1, 7, 11]},
                    1: {"HeiChole": [15, 18, 23]}
                }
            }
    elif args.data_split_type == "cholec-nct":
        # only cholec, with the surgery split corresponding to the one used at the nct
        if args.mode == "centralized":
            data_split = {
                "train": {
                    0: {"Cholec80": [2, 4, 6, 12, 24, 29, 34, 37, 38, 39, 44, 58, 60, 61, 64, 66, 75, 78, 79, 80] +
                        [1, 3, 5, 9, 13, 16, 18, 21, 22, 25, 31, 36, 45, 46, 48, 50, 62, 71, 72, 73] +
                        [10, 15, 17, 20, 32, 41, 42, 43, 47, 49, 51, 52, 53, 55, 56, 69, 70, 74, 76, 77]}
                },

                "test": {
                    0: {"Cholec80": [7, 8, 19, 23, 26, 33, 67]},
                    1: {"Cholec80": [11, 27, 30, 54, 63, 68]},
                    2: {"Cholec80": [14, 28, 35, 40, 57, 59, 65]}                }
            }
        else:  # federated training
            data_split = {
                "train": {
                    0: {"Cholec80": [4, 12, 13, 24, 25, 37, 45, 46, 47, 48, 49, 53, 70, 72, 73, 76]},
                    1: {"Cholec80": [1, 2, 5, 6, 9, 15, 16, 17, 29, 32, 36, 41, 42, 43, 50, 51, 58, 61, 62, 66, 69, 77,
                                     78,  39, 43, 52, 55, 56, 74, 79]},
                    2: {"Cholec80": [3, 10, 18, 20, 21, 22, 31, 34, 38, 44, 60, 64, 71, 75, 80]}
                },
                "test": {
                    0: {"Cholec80": [7, 8, 19, 23, 26, 33, 67]},
                    1: {"Cholec80": [11, 27, 30, 54, 63, 68]},
                    2: {"Cholec80": [14, 28, 35, 40, 57, 59, 65]}
                }
            }
    elif args.data_split_type == "quick-test":
        if args.mode == "centralized":
            data_split = {
                "train": {
                    0: {"HeiChole": list(range(4, 5)) + list(range(16, 17))}
                },
                "test": {
                    0: {"HeiChole": [1]},
                    1: {"HeiChole": [13]},
                }
            }
        else:  # federated training
            data_split = {
                "train": {
                    0: {"HeiChole": list(range(4, 5))},
                    1: {"HeiChole": list(range(16, 17))},
                },
                "test": {
                    0: {"HeiChole": [1]},
                    1: {"HeiChole": [13]},
                }
            }
    return data_split


def setup_client_dataloaders(args, data_split):
    """
    Create trainloaders for each client and a testloader
    :param args: config as in options.py
    :param data_split: dict for data assignment in the form
           {"train": {0: client0_train_data_indices, ..., n-1: clientn-1_train_data_indices},
            "test": {0: client0_test_data_indices, ..., n-1: clientn-1_test_data_indices}}
    :return: trainloaders, testloader
    """

    train_data_split = data_split["train"]

    if args.mode == "FedDG":
        global_amplitude_bank = {}

        for client_idx, client_data in train_data_split.items():
            for dataset_name, surgery_names in client_data.items():
                if client_idx not in global_amplitude_bank.keys():
                    global_amplitude_bank[client_idx] = ([], dataset_name)

                for name in surgery_names:
                    if dataset_name == "HeiChole":
                        global_amplitude_bank[client_idx][0].extend(
                            glob.glob(f"{args.data_root}/train/HeiChole/{str(name)}/amplitudes/*")
                        )
                    elif dataset_name == "Cholec80":
                        global_amplitude_bank[client_idx][0].extend(
                            glob.glob(f"{args.data_root}/train/Cholec80/{str(name)}/amplitudes/*")
                        )

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # normalization for images

    # train dataloaders
    trainloaders = {}
    for client_idx, client_data in train_data_split.items():
        print(f"train {client_idx}")

        if args.mode == "FedDG":
            freq_interpolation = AmplitudeInterpolation(
                client_idx, global_amplitude_bank
            )
            transform = transforms.Compose(
                [freq_interpolation, transforms.ToTensor(), normalize]
            )  # applies normalization and wraps into tensor
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(), normalize])  # applies normalization and wraps into tensor

        trainset = Cohort(
            args.data_root,  # path to the data
            client_data,  # list of surgeries to be contained
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
        trainloaders[client_idx] = trainloader

    # test dataloader
    testloaders = {}
    test_data_split = data_split["test"]

    if args.mode == "FedDG":  # remove FedDG augmentation for the testset
        transform = transforms.Compose(
            [transforms.ToTensor(), normalize])  # applies normalization and wraps into tensor

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

    return trainloaders, testloaders


def setup_optimizer_and_scheduler(args, model, scope, len_trainloader=None):
    assert scope == "global" or scope == "local"
    if scope == "global":
        optimizer_type = args.global_optimizer
        initial_lr = args.global_lr
    elif scope == "local":
        optimizer_type = args.local_optimizer
        initial_lr = args.local_lr

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
        # weight decay 1e-5, momentum 0.9
    elif optimizer_type == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=initial_lr)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)

    lr_scheduler = None
    if args.lr_scheduler_type == "multiplicative":
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: args.lr_decay  # 1/np.sqrt(epoch)
        )
    elif args.lr_scheduler_type == "onecycle" and len_trainloader is not None:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,  # set optimizer to be scheduled
            max_lr=initial_lr,  # set max lr = initial lr
            epochs=args.n_global_epochs,  # define number of epochs
            steps_per_epoch=len_trainloader,  # define number of batches per epoch
        )
    return optimizer, lr_scheduler


def setup_model(model_name, num_classes):
    if model_name == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
        # reconfigure num classes
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        # reconfigure num classes
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        # reconfigure num classes
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnext50":
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        # reconfigure num classes
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "convnext_tiny":
        model = torchvision.models.convnext_tiny(pretrained=True)
        # reconfigure num classes
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
        # reconfigure num classes
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "vgg13":
        model = torchvision.models.vgg13(pretrained=True)
        # reconfigure num classes
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
        # reconfigure num classes
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)
        # reconfigure num classes
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def compute_pos_weight(dataloader):
    """
    Compute fraction of positive / negative labels as pos_weight for the loss.
    """
    targets = []
    for _, _, target in dataloader:
        targets.append(target.numpy())
    targets = np.concatenate(targets)
    sum_targets = np.sum(targets, axis=0)
    pos_weight = (targets.shape[0] - sum_targets) / sum_targets
    for i in range(len(pos_weight)):
        if pos_weight[i] == np.inf:
            pos_weight[i] = 1
    print(pos_weight)
    return torch.Tensor(pos_weight)


def send_message(connection, text):
    (comm, url, chat_id) = connection
    if comm:
        try:
            params = {"chat_id": chat_id, "text": text}
            requests.post(url, params=params)
        except:
            logging.error(traceback.format_exc())


def test_inference(model, testloaders, criterion):
    """
    Compute test performance of the model for all testloaders combined.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # define GPU is available, else use CPU

    sigmoid = torch.nn.Sigmoid()

    model.to(device)
    model.eval()

    losses_global = []
    predictions_global = []
    targets_global = []

    # only compute client-wise results if there are multiple testloaders
    if len(testloaders) > 1:
        local_weighted_results = []
        local_macro_results = []
        local_individual_results = []
    else:
        local_weighted_results = None
        local_macro_results = None
        local_individual_results = None

    with torch.inference_mode():  # reduce workload by disabling gradient calculation

        for client_idx, testloader in testloaders.items():
            losses_local = []
            predictions_local = []
            targets_local = []
            for _, input, target in tqdm(testloader, desc=f"Test Client {client_idx}", leave=False):
                input = input.to(device)
                output = model(input)
                pred = (sigmoid(output) >= 0.5).detach().cpu().numpy()
                loss = criterion(output, target.to(device).float())

                predictions_global.extend(pred)
                targets_global.extend(target.numpy())
                losses_global.append(loss.item())

                if len(testloaders) > 1:
                    predictions_local.extend(pred)
                    targets_local.extend(target.numpy())
                    losses_local.append(loss.item())

            if len(testloaders) > 1:
                # weighted scores
                precision, recall, f1, _ = precision_recall_fscore_support(targets_local, predictions_local,
                                                                           average='weighted', zero_division=0)
                local_weighted_results.append([precision, recall, f1, np.mean(losses_local)])

                # macro scores
                precision, recall, f1, _ = precision_recall_fscore_support(targets_local, predictions_local,
                                                                           average='macro', zero_division=0)
                local_macro_results.append([precision, recall, f1, np.mean(losses_global)])

                # individual scores
                precision, recall, f1, _ = precision_recall_fscore_support(targets_local, predictions_local,
                                                                           average=None, zero_division=0)
                local_individual_results.append([precision, recall, f1, np.mean(losses_local)])

    # weighted scores
    precision, recall, f1, _ = precision_recall_fscore_support(targets_global, predictions_global,
                                                               average='weighted', zero_division=0)
    global_weighted_results = [precision, recall, f1, np.mean(losses_global)]

    # macro scores
    precision, recall, f1, _ = precision_recall_fscore_support(targets_global, predictions_global,
                                                               average='macro', zero_division=0)
    global_macro_results = [precision, recall, f1, np.mean(losses_global)]

    # individual scores
    precision, recall, f1, _ = precision_recall_fscore_support(targets_global, predictions_global,
                                                               average=None, zero_division=0)
    global_individual_results = [precision, recall, f1, np.mean(losses_global)]

    model.train()
    model.to(torch.device("cpu"))
    return global_weighted_results, global_macro_results, global_individual_results, \
           local_weighted_results, local_macro_results, local_individual_results
