from src.dataset.CIFAR10 import CIFAR10
import src.utils.Log as Log

def data_loader(data_name=None, batch_size=None, distribution=None, train=True):
    if data_name == 'CIFAR10':
        data = CIFAR10(batch_size, distribution, train)
    else:
        data = None
        Log.print_with_color("Incorrect name data !!!", "yellow")
    return data