import time
import pickle
import copy

import src.utils.Log as Log
from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from src.train.VGG16 import Train_VGG16
from src.dataset.dataloader import data_loader

class RpcClient:
    def __init__(self, client_id, channel, device):
        self.client_id = client_id
        self.channel = channel
        self.model_train = None
        self.train_loader = None
        self.device = device

        self.response = None
        self.model = None
        self.label_count = None

    def wait_response(self):
        status = True
        reply_queue_name = f'reply_{self.client_id}'
        self.channel.queue_declare(reply_queue_name, durable=False)
        while status:
            method_frame, header_frame, body = self.channel.basic_get(queue=reply_queue_name, auto_ack=True)
            if body:
                status = self.response_message(body)
            time.sleep(0.5)

    def response_message(self, body):
        self.response = pickle.loads(body)
        Log.print_with_color(f"[<<<] Client received: {self.response['message']}", "blue")
        action = self.response["action"]
        state_dict = self.response["parameters"]

        if action == "START":
            label_count = self.response['label_count']
            learning = self.response['learning']
            data_name = self.response["data_name"]

            self.model_train = Train_VGG16(self.client_id, self.channel, self.device)

            self.label_count = label_count
            Log.print_with_color(f"Label distribution of client: {self.label_count}", "yellow")

            # Load model
            if self.model is None:
                klass = VGG16_CIFAR10
                self.model = klass()

            # Read parameters and load to model
            if state_dict:
                self.model.load_state_dict(state_dict)

            self.model.to(self.device)

            # Start training
            if self.train_loader is None:
                self.train_loader = data_loader(data_name, learning['batch-size'], self.label_count, train=True)

            result, size = self.model_train.train_on_device(self.model, learning, self.train_loader)

            # Stop training, then send parameters to server
            model_state_dict = copy.deepcopy(self.model.state_dict())
            if self.device != "cpu":
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to('cpu')
            data = {"action": "UPDATE", "client_id": self.client_id,
                    "result": result, "size": size,
                    "message": "Sent parameters to Server", "parameters": model_state_dict}
            Log.print_with_color("[>>>] Client sent parameters to server", "red")
            self.send_to_server(data)
            return True
        elif action == "STOP":
            return False

    def send_to_server(self, message):
        self.response = None

        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

        return self.response
