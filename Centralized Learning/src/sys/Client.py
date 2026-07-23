import torch
import random
import numpy as np
import src.utils.Log as Log
from src.train.VGG16 import train_on_device
from src.dataset.dataloader import data_loader

from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from src.val.get_val import get_val

class Client:
    def __init__(self, config):
        self.model_name = config["server"]["model"]
        self.data_name = config["server"]["data-name"]
        self.global_round = config["server"]["global-round"]
        self.round = self.global_round
        self.save_parameters = config["server"]["parameters"]["save"]
        self.load_parameters = config["server"]["parameters"]["load"]
        self.validation = config["server"]["validation"]

        # Clients
        self.learning = config["learning"]
        self.data_distribution = config["server"]["data-distribution"]

        # Data distribution
        self.non_iid = self.data_distribution["non-iid"]
        self.num_label = self.data_distribution["num-label"]
        self.num_sample = self.data_distribution["num-sample"]
        self.label_counts = None

        random.seed(42)

        log_path = config["log_path"]

        self.responses = {}  # Save response
        self.round_result = True
        debug_mode = config["debug_mode"]
        self.logger = Log.Logger(f"{log_path}/app.log", debug_mode)
        Log.print_with_color(f"Application start.", "green")

    def distribution(self):
        if self.non_iid:
            label_distribution = np.array([[0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]])
            self.label_counts = (label_distribution * self.num_sample).astype(int)
        else:
            self.label_counts = np.full((1, self.num_label), self.num_sample // self.num_label)

    def start(self):
        self.distribution()
        model = VGG16_CIFAR10()
        train_data = data_loader(self.data_name, self.learning['batch-size'], self.label_counts[0], True)

        for _ in range(self.round):
            model = train_on_device(model, self.learning, train_data, 'cpu')

            if not get_val(self.model_name, self.data_name, model, self.logger):
                break




