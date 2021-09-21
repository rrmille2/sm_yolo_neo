import torch
import os
import logging
import base64
import io
import json
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import traceback

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
JPEG_CONTENT_TYPE = 'application/x-image'
logger = logging.getLogger()
DEBUG = os.getenv("PROFILE_METHODS")

def model_fn(model_dir):
    import os
    nvidia_smi = "nvidia-smi"
    cuda_dir = "/usr/local/cuda"
    try:
        print(f"Environment is {os.environ}\n")
        print(f"Listing cuda directory . {os.listdir(cuda_dir)}\n")
        print(f"Nvidia SMI - {os.system(nvidia_smi)}\n")
    except Exception as e:
        logging.warning("Invoking custom service failed.", exc_info=True)
        pass
                              
                                      
    traceback.print_stack()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'COMPILEDMODEL' in os.environ and os.environ['COMPILEDMODEL'] == 'True':
        import neopytorch
        logger.info('using compiled model')
        neopytorch.config(model_dir=model_dir, neo_runtime=True)
        model = torch.jit.load('compiled.pt', map_location=device)
        return model.to(device)

    # Model
    logger.info('using uncompiled model')
    model = torch.jit.load('model.pth', map_location=device)
    model.to(device)
    return model 

def predict_fn(image, model):
    # traceback.print_stack()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    output = model.forward(image)
    return output

def input_fn(request_body, content_type):
    # traceback.print_stack()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if content_type == JPEG_CONTENT_TYPE:
        iobytes = io.BytesIO(request_body)
        decoded = Image.open(iobytes)
        preprocess = transforms.Compose([
            transforms.Resize(416),
            transforms.CenterCrop(416),
            transforms.ToTensor(),
            transforms.Normalize(mean=[
                0.485, 0.456, 0.406],
                std=[ 0.229, 0.224, 0.225])
            ])
        normalized = preprocess(decoded)
        batchified = normalized.unsqueeze(0)
        return batchified.to(device)

    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def output_fn(data, accept):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    buffer.seek(0)
    return buffer.read()