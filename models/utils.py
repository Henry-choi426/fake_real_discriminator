from models.timm_models import EffNet, VIT

def get_model(model_name:str, model_args:dict):
    if model_name == 'effnet':
        return EffNet(**model_args)
    
    elif model_name == 'vit':
        return VIT(**model_args)
    else: 
        raise ValueError(f'Model name {model_name} is not valid.')

if __name__ == '__main__':
    pass