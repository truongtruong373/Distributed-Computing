import argparse
from src.sys.Client import Client
import yaml

parser = argparse.ArgumentParser(description="Train in only on device.")

args = parser.parse_args()

with open('config.yaml') as file:
    config = yaml.safe_load(file)

if __name__ == "__main__":
    server = Client(config)
    server.start()
