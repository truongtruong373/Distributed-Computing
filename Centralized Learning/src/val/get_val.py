from src.val.VGG16 import val_VGG16

def get_val(model_name, data_name, model, logger):
    if model_name == 'VGG16':
        return val_VGG16(data_name, model, logger)
    else:
        return False

