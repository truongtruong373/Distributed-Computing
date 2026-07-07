# Distributed Computing
![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![RabbitMQ](https://img.shields.io/badge/RabbitMQ-FF6600?style=for-the-badge&logo=rabbitmq&logoColor=white)


This repository provides implementations and experiments related to **distributed computing for machine learning**, focusing on two major paradigms: **Federated Learning** and **Split Learning**.

The main goal of this project is to study how machine learning models can be trained across multiple distributed devices while reducing computation cost, communication overhead, and data privacy risks. Instead of training a model on a single centralized machine, the system allows multiple clients, edge devices, or servers to collaboratively participate in the training process.

## Overview
Distributed machine learning has become an important approach for modern AI systems, especially in scenarios where data is generated across many devices such as mobile phones, IoT devices, edge servers, and cloud platforms. In these environments, collecting all data into one central server may be inefficient, expensive, or unsuitable due to privacy constraints.

This project explores different distributed training architectures, including:
- **Federated Learning**, where clients train local models using their own data and only share model updates with a central server.
- **Split Learning**, where a neural network is divided into client-side and server-side parts, allowing clients to perform only part of the computation while the server completes the remaining training process.

Through these implementations, the repository aims to provide a clear understanding of how distributed learning systems operate, how communication between clients and servers is handled, and how different architectures affect training performance.

## Project Structure

```text
Distributed-Computing/
│
├── Federated Learning/   # Federated Learning architectures and experiments
├── Split Learning/       # Split Learning architectures and experiments
├── requirements.txt      # Required Python packages
├── LICENSE
└── README.md