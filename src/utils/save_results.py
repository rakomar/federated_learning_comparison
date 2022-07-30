import shutil
import os
import csv

import torch
import numpy as np


def clear_directory(path):
    """
    Removes and adds a new directory with given path.
    :param path: path to directory
    """
    try:
        shutil.rmtree(path)
    except OSError as er:
        print(er)
        print("Creating new directory.\n")
    os.mkdir(path)


def create_header_names(term, num_clients=None, labels=None):
    if num_clients is not None and labels is not None:
        return [f"{i}-{term}-{tool}" for i in range(num_clients) for tool in labels]
    if num_clients is not None:
        return [f"{i}-{term}" for i in range(num_clients)]
    if labels is not None:
        return [f"{term}-{tool}" for tool in labels]


class ResultWriter:
    def __init__(self, args, results_path, writer_type, labels, num_clients=None):
        assert writer_type == "train" or writer_type == "test"
        self.path = f"{results_path}/{writer_type}"
        clear_directory(self.path)
        self.num_clients = num_clients
        self.epoch = 0
        self.best_f1 = 0
        self.performance_history = None
        self.threshold = args.criteria_threshold

        # create random_indices file
        file = open(f"{results_path}/options.txt", "w")
        file.write("# options (args) used for training \n\n")
        file.write(str(args))
        file.close()

        # f1 scores + losses
        # if args.mode == "centralized":
        #     if writer_type == "train":
        #         self.loss_writer = LossWriter(self.path)  # to track server loss
        #         self.global_weighted_f1score_writer = GlobalWeightedScoresWriter(self.path)
        #         self.global_individual_f1score_writer = GlobalIndividualScoresWriter(self.path, labels)
        #     elif writer_type == "test":
        #         # global writers track server loss
        #         self.global_weighted_f1score_writer = GlobalWeightedScoresWriter(self.path)
        #         self.global_individual_f1score_writer = GlobalIndividualScoresWriter(self.path, labels)
        if args.mode == "centralized" or args.mode == "local_only":
            if writer_type == "train":
                self.loss_writer = LossWriter(self.path)  # to track server loss
                self.global_weighted_f1score_writer = GlobalWeightedScoresWriter(self.path)
                self.global_macro_f1score_writer = GlobalMacroScoresWriter(self.path)
                self.global_individual_f1score_writer = GlobalIndividualScoresWriter(self.path, labels)
            elif writer_type == "test":
                # global writers track server loss
                self.global_weighted_f1score_writer = GlobalWeightedScoresWriter(self.path)
                self.global_macro_f1score_writer = GlobalMacroScoresWriter(self.path)
                self.global_individual_f1score_writer = GlobalIndividualScoresWriter(self.path, labels)
                self.local_weighted_f1score_writer = LocalWeightedScoresWriter(self.path, num_clients)
                self.local_macro_f1score_writer = LocalMacroScoresWriter(self.path, num_clients)
                self.local_individual_f1score_writer = LocalIndividualScoresWriter(self.path, num_clients, labels)
        else:  # federated training
            if writer_type == "train":
                self.loss_and_grads_writer = LossAndGradsWriter(self.path, num_clients)  # to track server loss
                self.local_weighted_f1score_writer = LocalWeightedScoresWriter(self.path, num_clients)
                self.local_macro_f1score_writer = LocalMacroScoresWriter(self.path, num_clients)
                self.local_individual_f1score_writer = LocalIndividualScoresWriter(self.path, num_clients, labels)
            elif writer_type == "test":
                # global writers track server loss
                self.global_weighted_f1score_writer = GlobalWeightedScoresWriter(self.path)
                self.global_macro_f1score_writer = GlobalMacroScoresWriter(self.path)
                self.global_individual_f1score_writer = GlobalIndividualScoresWriter(self.path, labels)
                self.local_weighted_f1score_writer = LocalWeightedScoresWriter(self.path, num_clients)
                self.local_macro_f1score_writer = LocalMacroScoresWriter(self.path, num_clients)
                self.local_individual_f1score_writer = LocalIndividualScoresWriter(self.path, num_clients, labels)

        # inference models
        if writer_type == "test":
            self.models_path = f"{self.path}/models"
            clear_directory(self.models_path)

    def increment_epoch(self):
        self.epoch += 1

    def write_loss(self, loss):
        self.loss_writer.write(self.epoch, loss)

    def write_loss_and_grads(self, loss, grads):
        self.loss_and_grads_writer.write(self.epoch, loss, grads)

    def write_weighted_f1score(self, scope, f1score_results):
        assert scope == "global" or scope == "local"
        if scope == "global":
            self.global_weighted_f1score_writer.write(self.epoch, f1score_results)
        elif scope == "local":
            self.local_weighted_f1score_writer.write(self.epoch, f1score_results)

    def write_individual_f1score(self, scope, f1score_results):
        assert scope == "global" or scope == "local"
        if scope == "global":
            self.global_individual_f1score_writer.write(self.epoch, f1score_results)
        elif scope == "local":
            self.local_individual_f1score_writer.write(self.epoch, f1score_results)

    def write_all_f1scores(self, global_weighted, global_macro, global_individual, local_weighted, local_macro,
                           local_individual):
        self.global_weighted_f1score_writer.write(self.epoch, global_weighted)
        self.global_macro_f1score_writer.write(self.epoch, global_macro)
        self.global_individual_f1score_writer.write(self.epoch, global_individual)
        self.local_weighted_f1score_writer.write(self.epoch, local_weighted)
        self.local_macro_f1score_writer.write(self.epoch, local_macro)
        self.local_individual_f1score_writer.write(self.epoch, local_individual)

    def save_best_model_if_improved(self, model, model_performance):
        # check if the global test performance is improved, then save model
        if model_performance > self.best_f1:
            self.best_f1 = model_performance
            torch.save(model, f"{self.models_path}/best")

    def save_model_if_stopping_criteria_met(self, model, model_performance):
        # hard coded for 5 history entries
        num_history_entries = 5
        if self.performance_history is None:
            self.performance_history = [model_performance]
        elif len(self.performance_history) < num_history_entries:

            self.performance_history.insert(0, model_performance)
        else:
            self.performance_history.insert(0, model_performance)
            self.performance_history.pop()

        criteria_met = True
        for i in range(1, len(self.performance_history)):
            if abs(self.performance_history[i-1] - self.performance_history[i]) > self.threshold:
                criteria_met = False
                break
        if criteria_met:
            torch.save(model, f"{self.models_path}/criteria_met")
        return criteria_met

    def save_final_model(self, model):
        torch.save(model, f"{self.models_path}/final")

    # def close_writer(self):
    #     for writer in self.writers:
    #         writer.file.close()


class GlobalWeightedScoresWriter:
    """
    Report results for combined dataset (all dataloaders)
    """
    def __init__(self, path):
        self.path = path
        with open(f"{path}/GlobalWeightedScores.csv", "w", newline='') as file:
            writer = csv.writer(file)
            header = ["Epoch", "Precision", "Recall", "F1Score", "Loss"]
            writer.writerow(header)

    def write(self, epoch, f1score_results):
        with open(f"{self.path}/GlobalWeightedScores.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch] + f1score_results)


class GlobalMacroScoresWriter:
    """
    Report results for combined dataset (all dataloaders)
    """
    def __init__(self, path):
        self.path = path
        with open(f"{path}/GlobalMacroScores.csv", "w", newline='') as file:
            writer = csv.writer(file)
            header = ["Epoch", "Precision", "Recall", "F1Score", "Loss"]
            writer.writerow(header)

    def write(self, epoch, f1score_results):
        with open(f"{self.path}/GlobalMacroScores.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch] + f1score_results)


class GlobalIndividualScoresWriter:
    """
    Report results for each tool individually (average=None)
    """
    def __init__(self, path, labels):
        self.path = path
        with open(f"{path}/GlobalIndividualScores.csv", "w", newline='') as file:
            writer = csv.writer(file)
            header = [
                "Epoch",
                *create_header_names("Precision", labels=labels),
                *create_header_names("Recall", labels=labels),
                *create_header_names("F1Score", labels=labels),
                "Loss"
            ]
            writer.writerow(header)

    def write(self, epoch, f1score_results):
        with open(f"{self.path}/GlobalIndividualScores.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [epoch] +
                list(f1score_results[0]) +
                list(f1score_results[1]) +
                list(f1score_results[2]) +
                [f1score_results[3]]
            )


class LocalWeightedScoresWriter:
    """
    Report results for each client dataset individually (single dataloaders)
    """
    def __init__(self, path, num_clients):
        self.path = path
        self.num_clients = num_clients
        with open(f"{path}/LocalWeightedScores.csv", "w", newline='') as file:
            writer = csv.writer(file)
            header = [
                "Epoch",
                *create_header_names("Precision", num_clients=num_clients),
                *create_header_names("Recall", num_clients=num_clients),
                *create_header_names("F1Score", num_clients=num_clients),
                *create_header_names("Loss", num_clients=num_clients)
            ]
            writer.writerow(header)

    def write(self, epoch, f1score_results):
        with open(f"{self.path}/LocalWeightedScores.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [epoch] +
                [f1score_results[i][0] for i in range(self.num_clients)] +
                [f1score_results[i][1] for i in range(self.num_clients)] +
                [f1score_results[i][2] for i in range(self.num_clients)] +
                [f1score_results[i][3] for i in range(self.num_clients)]
            )


class LocalMacroScoresWriter:
    """
    Report results for each client dataset individually (single dataloaders)
    """
    def __init__(self, path, num_clients):
        self.path = path
        self.num_clients = num_clients
        with open(f"{path}/LocalMacroScores.csv", "w", newline='') as file:
            writer = csv.writer(file)
            header = [
                "Epoch",
                *create_header_names("Precision", num_clients=num_clients),
                *create_header_names("Recall", num_clients=num_clients),
                *create_header_names("F1Score", num_clients=num_clients),
                *create_header_names("Loss", num_clients=num_clients)
            ]
            writer.writerow(header)

    def write(self, epoch, f1score_results):
        with open(f"{self.path}/LocalMacroScores.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [epoch] +
                [f1score_results[i][0] for i in range(self.num_clients)] +
                [f1score_results[i][1] for i in range(self.num_clients)] +
                [f1score_results[i][2] for i in range(self.num_clients)] +
                [f1score_results[i][3] for i in range(self.num_clients)]
            )


class LocalIndividualScoresWriter:
    """
    Report results for each client dataset and tool individually (single dataloaders, average=None)
    """
    def __init__(self, path, num_clients, labels):
        self.path = path
        self.num_clients = num_clients
        self.num_labels = len(labels)
        with open(f"{path}/LocalIndividualScores.csv", "w", newline='') as file:
            writer = csv.writer(file)
            header = [
                "Epoch",
                *create_header_names("Precision", num_clients=num_clients, labels=labels),
                *create_header_names("Recall", num_clients=num_clients, labels=labels),
                *create_header_names("F1Score", num_clients=num_clients, labels=labels),
                *create_header_names("Loss", num_clients=num_clients)
            ]
            print(header)
            print(len(header))
            writer.writerow(header)
            reader = csv.reader(file)
            # x = next(reader)
            # print(x)
            # print(len(x))

    def iterate_results(self, f1score_results, j):
        [list(f1score_results[i][j]) for i in range(self.num_clients)]

    def write(self, epoch, f1score_results):
        with open(f"{self.path}/LocalIndividualScores.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [epoch] +
                [f1score_results[i][0][j] for i in range(self.num_clients) for j in range(self.num_labels)] +
                [f1score_results[i][1][j] for i in range(self.num_clients) for j in range(self.num_labels)] +
                [f1score_results[i][2][j] for i in range(self.num_clients) for j in range(self.num_labels)] +
                [f1score_results[i][3] for i in range(self.num_clients)]
            )


class LossWriter:
    """
    Report only loss (used for global training results)
    """
    def __init__(self, path):
        self.path = path
        with open(f"{path}/Loss.csv", "w", newline='') as file:
            writer = csv.writer(file)
            header = ["Epoch", "Loss"]
            writer.writerow(header)

    def write(self, epoch, loss):
        with open(f"{self.path}/Loss.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, loss])


class LossAndGradsWriter:
    """
    Report loss and client grads (used for server training results)
    """
    def __init__(self, path, num_clients):
        self.path = path
        self.num_clients = num_clients
        with open(f"{path}/LossAndGrads.csv", "w", newline='') as file:
            writer = csv.writer(file)
            header = [
                "Epoch", "Loss",
                "AvgGradLengthGlobalLocal",
                *create_header_names("GradLengthGlobalLocal", num_clients=num_clients),
                "AvgGradLengthLocal",
                *create_header_names("GradLengthLocal", num_clients=num_clients),
                "AvgGradCosinusSimilaritiesGlobalLocal",
                *create_header_names("GradCosinusSimilaritiesGlobalLocal", num_clients=num_clients),
                *create_header_names("AvgGradCosinusSimilaritiesLocalLocal", num_clients=num_clients),
                *create_header_names("GradCosinusSimilaritiesLocalLocal",
                                     num_clients=num_clients, labels=list(range(num_clients)))
            ]
            writer.writerow(header)

    def write(self, epoch, loss, grads):
        with open(f"{self.path}/LossAndGrads.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [epoch, loss] +
                [np.mean(grads[0])] +
                grads[0] +
                [np.mean(grads[1])] +
                grads[1] +
                [np.mean(grads[2])] +
                grads[2] +
                [np.mean(grads[3][i])-(1/self.num_clients) for i in range(self.num_clients)] +  # -(1/self.num_clients) because cos sim to itself (=1) included
                [grads[3][i][j] for i in range(self.num_clients) for j in range(self.num_clients)]
            )


