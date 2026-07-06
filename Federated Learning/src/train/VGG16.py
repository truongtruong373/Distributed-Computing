import time
import pickle
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

import src.utils.Log as Log

class Train_VGG16:
    def __init__(self, client_id, channel, device):
        self.client_id = client_id
        self.channel = channel
        self.device = device
        self.data_count = 0

    def send_to_server(self, message):
        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

    def train_on_device(self, model, learning, train_loader=None):
        optimizer = optim.SGD(model.parameters(), lr=learning['learning-rate'], momentum=learning['momentum'])
        criterion = nn.CrossEntropyLoss()

        model.to(self.device)
        data_iter = iter(train_loader)

        with tqdm(total=len(train_loader), desc="Processing", unit="step") as pbar:
            while True:
                # Training model
                model.train()
                optimizer.zero_grad()

                try:
                    training_data, labels = next(data_iter)
                    training_data = training_data.to(self.device)
                    output = model(training_data)
                    loss = criterion(output, labels)
                    if torch.isnan(loss).any():
                        Log.print_with_color("NaN detected in loss", "yellow")
                        result = False
                    loss.backward()

                    optimizer.step()
                    self.data_count += 1
                    # tqdm bar
                    pbar.update(1)

                except StopIteration:
                    break

        notify_data = {"action": "NOTIFY", "client_id": self.client_id, "message": "Finish training!"}

        # Finish epoch training, send notify to server
        Log.print_with_color("[>>>] Finish training!", "red")
        self.send_to_server(notify_data)

        broadcast_queue_name = f'reply_{self.client_id}'
        while True:  # Wait for broadcast
            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:
                received_data = pickle.loads(body)
                Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                if received_data["action"] == "PAUSE":
                    return result , self.data_count
            time.sleep(0.5)