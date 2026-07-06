import torch
import os
import random
import pika
import pickle
import sys
import numpy as np
import copy
import src.utils.Log as Log
import src.utils.Utils

from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from src.val.get_val import get_val

class Server:
    def __init__(self, config):
        # RabbitMQ
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model"]
        self.data_name = config["server"]["data-name"]
        self.total_clients = config["server"]["clients"]
        self.global_round = config["server"]["global-round"]
        self.round = self.global_round
        self.save_parameters = config["server"]["parameters"]["save"]
        self.load_parameters = config["server"]["parameters"]["load"]
        self.validation = config["server"]["validation"]
        self.cut_layers = config["server"]["cut-layers"]

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

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.current_clients = [0 for _ in range(len(self.total_clients))]
        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.first_layer_clients = 0
        self.responses = {}  # Save response
        self.list_clients = []
        self.round_result = True

        self.global_params = None
        self.global_sizes = None
        self.avg_state_dict = []

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug_mode"]
        self.logger = Log.Logger(f"{log_path}/app.log", debug_mode)
        Log.print_with_color(f"Application start. Server is waiting for {self.total_clients} clients.", "green")

    def distribution(self):
        if self.non_iid:
            label_distribution = np.array([[0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                           [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                           [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.1],
                                           [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.1],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1],
                                           ])
            self.label_counts = (label_distribution * self.num_sample).astype(int)
        else:
            self.label_counts = np.full((self.total_clients[0], self.num_label), self.num_sample // self.num_label)

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]
        self.responses[routing_key] = message

        if action == "REGISTER":
            if (str(client_id), layer_id) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id))

            src.utils.Log.print_with_color(f"[<<<] Received message from client {client_id}", "blue")
            # Save messages from clients
            self.register_clients[layer_id - 1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:
                src.utils.Log.print_with_color("All clients are connected. Sending notifications.", "green")

                self.distribution()
                self.label_counts = self.label_counts.tolist()
                for idx, (client_id, layer_id) in enumerate(self.list_clients):
                    if layer_id == 1:
                        self.list_clients[idx] = (
                            client_id, layer_id, self.label_counts.pop())
                    else:
                        self.list_clients[idx] = (client_id, layer_id, [])

                self.global_params = [[] for _ in range(len(self.total_clients))]
                self.global_sizes = [[] for _ in range(len(self.total_clients))]
                src.utils.Log.print_with_color(f'Cut point: {self.cut_layers}', 'yellow')
                self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                self.notify_clients()

        elif action == "NOTIFY":
            src.utils.Log.print_with_color("[<<<] Received message from client: {message}", "blue")
            message = {"action": "PAUSE",
                       "message": "Pause training and please send your parameters",
                       "parameters": None}
            if layer_id == 1:
                self.first_layer_clients += 1

            if self.first_layer_clients == self.total_clients[0]:
                self.first_layer_clients = 0
                src.utils.Log.print_with_color(f"Received finish training notification", "yellow")

                for (client_id, layer_id, _) in self.list_clients:
                    self.send_to_response(client_id, pickle.dumps(message))

        elif action == "UPDATE":
            data_message = message["message"]
            result = message["result"]
            src.utils.Log.print_with_color(f"[<<<] Received message from {client_id}: {data_message}", "blue")

            self.current_clients[layer_id - 1] += 1
            if not result:
                self.round_result = False

            # Save client's model parameters
            if self.save_parameters and self.round_result:
                model_state_dict = message["parameters"]
                client_size = message["size"]
                self.global_params[layer_id - 1].append(model_state_dict)
                self.global_sizes[layer_id - 1].append(client_size)

            # If consumed all client's parameters
            if self.current_clients == self.total_clients:
                src.utils.Log.print_with_color("Collected all parameters.", "yellow")
                if self.save_parameters and self.round_result:
                    self.avg_params()
                    self.global_params = [[] for _ in range(len(self.total_clients))]
                    self.global_sizes = [[] for _ in range(len(self.total_clients))]
                self.current_clients = [0 for _ in range(len(self.total_clients))]
                # Test
                if self.save_parameters and self.validation and self.round_result:
                    state_dict_full = self.concatenate()
                    self.avg_state_dict = []
                    if not get_val(self.model_name, self.data_name, state_dict_full, self.logger):
                        self.logger.log_warning("Training failed!")
                        self.round = 0
                    else:
                        # Save to files
                        torch.save(state_dict_full, f'{self.model_name}_{self.data_name}.pth')
                        self.round -= 1
                else:
                    self.round -= 1

                # Start a new training round
                self.round_result = True

                if self.round > 0:
                    self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                    self.notify_clients()
                else:
                    self.logger.log_info("Stop training !!!")
                    self.notify_clients(start=False)
                    sys.exit()

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, start=True):
        # Send message to clients when consumed all clients
        for (client_id, layer_id, label) in self.list_clients:
            # Read parameters file
            filepath = f'{self.model_name}_{self.data_name}.pth'
            state_dict = None

            if start:
                if layer_id == 1:
                    layers = [0, self.cut_layers[0]]
                else:
                    layers = [self.cut_layers[-1], -1]

                if self.load_parameters:
                    if os.path.exists(filepath):
                        full_state_dict = torch.load(filepath, weights_only=True)
                        klass = VGG16_CIFAR10
                        if layer_id == 1:
                            model = klass(end_layer=layers[1])
                        else:
                            model = klass(start_layer=layers[0])

                        state_dict = model.state_dict()
                        keys = state_dict.keys()

                        for key in keys:
                            state_dict[key] = full_state_dict[key]

                        Log.print_with_color(f"Load model successfully", "green")

                    else:
                        self.logger.log_info(f"File {filepath} does not exist.")

                src.utils.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")

                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "parameters": copy.deepcopy(state_dict),
                            "num_layers": len(self.total_clients),
                            "layers": layers,
                            "model_name": self.model_name,
                            "data_name": self.data_name,
                            "learning": self.learning,
                            "label_count": label}
                self.send_to_response(client_id, pickle.dumps(response))

            else:
                src.utils.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                response = {"action": "STOP",
                            "message": "Stop training!",
                            "parameters": None}
                self.send_to_response(client_id, pickle.dumps(response))

    def start(self):
        self.channel.start_consuming()

    def send_to_response(self, client_id, message):
        reply_queue_name = f'reply_{client_id}'
        self.reply_channel.queue_declare(reply_queue_name, durable=False)

        src.utils.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def avg_params(self):
        layer_sizes = self.global_sizes
        layer_params = self.global_params

        for layer_idx, list_state_dicts in enumerate(layer_params):
            list_sizes = layer_sizes[layer_idx]
            if not list_state_dicts or not list_sizes:
                self.avg_state_dict.append({})
                continue
            avg_sd = src.utils.Utils.fedavg_state_dicts(list_state_dicts, weights=list_sizes)
            self.avg_state_dict.append(avg_sd)

    def concatenate(self):
        avg_layers = self.avg_state_dict
        full_dict = {}

        for idx, layer_dict in enumerate(avg_layers):
            sd = layer_dict
            full_dict.update(copy.deepcopy(sd))
        return full_dict
