from models.timm_models import timm_model

def get_model(model_name:str, model_args:dict):
    if model_name != '':
        return timm_model(**model_args)
    else: 
        raise ValueError(f'Model name {model_name} is not valid.')

if __name__ == '__main__':
    pass