from src.val.VGG16 import val_VGG16

def get_val(model_name, data_name, state_dict_full, logger):
    if model_name == 'VGG16':
        return val_VGG16(data_name, state_dict_full, logger)
    else:
        return False

