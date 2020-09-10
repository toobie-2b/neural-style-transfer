import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from io import BytesIO

max_size = 512


def load_image(image_path, device, image_size):
    '''
    A function to load an image and convert it to tensor
    for the model.\n
    `image_path`: Path/Link to the image.\n
    `device`: Load the tensor on the specified device.\n
    `image_size`: Resize the image to the desired dimensions (optional)
    '''

    if 'https' in image_path:
        response = request.get(image_path)
        image = Image.open(response.content).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')

    if image_size is not None:
        size = image_size
    else:
        if max(image.size) > max_size:
            h,w = image.size
            size = (int(w/max(image.size) * max_size),
                int(h/max(image.size) * max_size))
        else:
            size = max_size

    ImageToTensor = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # only accepting the first 3 channels i.e. RGB
    image = ImageToTensor(image)[:3, :, :].unsqueeze(0).to(device)
    return image


def save_image(tensor, image_path):
    image = tensor.squeeze(0).detach().cpu().numpy()
    image = np.moveaxis(image, 0, 2)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = np.clip(image, 0, 1)
    image = Image.fromarray(np.uint8(image * 255))
    image.save(image_path)