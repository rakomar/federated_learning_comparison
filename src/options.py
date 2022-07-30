from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, help="path to data", required=True)
    parser.add_argument("--data_split_type", type=str, default="real",
                        choices=["real", "heichole", "cholec-nct", "quick-test"],
                        help="how to split data between clients -> see utils.general")
    parser.add_argument("--mode", type=str, default="FedAvg",
                        choices=["centralized", "local_only", "FedAvg", "FedProx", "FedDyn", "FedMix", "FedDG", "FedRep"],
                        help="name of the procedure that should be ran centralized/local_only/federated algorithm name")
    parser.add_argument("--model_name", type=str, default="alexnet",
                        choices=["alexnet", "resnet18", "resnet50", "resnext50", "convnext_tiny", "vgg11", "vgg13", "vgg16", "vgg19"],
                        help="name of ML model")
    parser.add_argument('--preprocess', dest='preprocess_data', action='store_true',
                        help="whether to extract images from the raw surgery videos before running")
    parser.set_defaults(preprocess_data=False)
    parser.add_argument("--width", type=int, default=240, help="width the images are resized to, max 280")
    parser.add_argument("--height", type=int, default=135, help="width the images are resized to, max 158")
    parser.add_argument('--load_to_ram', dest='load_to_ram', action='store_true',
                        help="load frames into ram or always load from the disk")
    parser.set_defaults(load_to_ram=False)
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                        help="pin memory in dataloader")
    parser.set_defaults(pin_memory=False)
    parser.add_argument("--n_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--lr_scheduler_type", type=str, default="onecycle",
                        choices=["multiplicative", "onecycle"], help="type of learning rate scheduler")
    parser.add_argument("--lr_decay", type=float, default=0.85, help="client learning rate decay")
    parser.add_argument("--criteria_threshold", type=float, default=1e-4, help="threshold to stop training if loss is"
                                                                               "stagnant below threshold for 5 epochs")
    parser.add_argument('--no_pos_weight', dest='pos_weight', action='store_false',
                        help="use pos_weight for criterion")
    parser.set_defaults(pos_weight=True)
    parser.add_argument("--n_classes", type=int, default=6, choices=range(1, 8),
                        help="number of instrument classes (first ones taken)")

    # server /centralized options
    parser.add_argument("--global_optimizer", type=str, default="adamw",
                        choices=["sgd", "adagrad", "adam", "adamw"], help="optimizer to train server with")
    parser.add_argument("--n_global_epochs", type=int, default=50, help="number of epochs (communication rounds)")
    parser.add_argument("--global_lr", type=float, default=1e-3, help="initial server learning rate")
    # parser.add_argument("--weight_decay", type=float, default=1e-5, help="sgd weight decay")
    # parser.add_argument("--momentum", type=float, default=0.9, help="sgd momentum")

    # federated learning options
    # client options
    parser.add_argument("--local_optimizer", type=str, default="sgd",
                        choices=["sgd", "adagrad", "adam", "adamw"], help="optimizer to train clients with")
    parser.add_argument("--frac", type=float, default=1., help="fraction of clients to perform updates: C")
    parser.add_argument("--n_local_epochs", type=int, default=1, help="client epochs: E")
    parser.add_argument("--local_batch_size", type=int, default=128, help="client batch size: B")
    parser.add_argument("--local_lr", type=float, default=1e-3, help="initial client learning rate")
    parser.add_argument('--no_weighted_aggregation', dest='weighted_aggregation', action='store_false',
                        help="aggregation weighted with data amounts")
    parser.set_defaults(weighted_aggregation=True)
    parser.add_argument('--no_own_implementation', dest='own_implementation', action='store_false',
                        help="use the own implementation of client updates")
    parser.set_defaults(own_implementation=True)

    # FedProx
    parser.add_argument("--mu", type=float, default=1e-2, help="hyperparameter for FedProx")

    # FedDyn
    parser.add_argument("--alpha", type=float, default=1e-2, help="hyperparameter for FedDyn")

    # FedRep
    parser.add_argument("--n_local_head_epochs", type=int, default=1, help="number of epochs for head updates, "
                                                                           "the rest of local epochs goes into "
                                                                           "representation updates")
    args = parser.parse_args()
    return args
