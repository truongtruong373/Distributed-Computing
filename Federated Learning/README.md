# Federated-Learning-Basic

## RabbitMQ

Set up a RabbitMQ server for message communication over the network environment. `docker-compose.yaml` file:
```text
version: '3'

services:
  rabbitmq:
    image: rabbitmq:management
    container_name: rabbitmq
    ports:
      - "5672:5672"   # RabbitMQ main port
      - "15672:15672" # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: user
      RABBITMQ_DEFAULT_PASS: password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
volumes:
  rabbitmq_data:
    driver: local
```
Then run the RabbitMQ container:
```commandline
docker-compose up -d
```

## Configuration
```yaml
name: Federated Learning
server:
  global-round: 1
  clients: 1
  model: VGG16
  data-name: CIFAR10
  parameters:
    load: True
    save: True
  validation: True
  data-distribution:
    non-iid: False
    num-sample: 5000
    num-label: 10

rabbit:
  address: 127.0.0.1
  username: admin
  password: admin
  virtual-host: /

log_path: .
debug_mode: True

learning:
  learning-rate: 0.01
  momentum: 0.5
  batch-size: 32
```

## How to run
Alter your configuration, you need to run the server to listen and control the request from clients.

### Server
```commandline
python3 server.py
```

### Client
```commandline
python3 client.py
```
Where :
* `--device`: cpu or cuda.

