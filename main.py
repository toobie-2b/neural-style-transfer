import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import argparse
from layers import layers, style_weights
from helper import load_image, save_image
from utils import get_features, gram_matrix


def get_arguments():
    '''
    A handy little function to accept the command line arguments
    for this script
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--content', required=True, type=str,
        help='Path/Link to the content image')
    parser.add_argument('--style', required=True, type=str,
        help='Path/Link to the style image')
    parser.add_argument('--result', required=True, type=str,
        help='Path to where the result should be saved')
    parser.add_argument('--size', required=False, type=int,
        default=None, help='Size of the generated stylized image')
    parser.add_argument('--steps', required=False, default=4000,
        type=int, help='Number of iterations')
    parser.add_argument('--alpha', required=False, default=1,
        type=float, help='Weight of the content loss')
    parser.add_argument('--beta', required=False, default=1e2,
        type=float, help='Weight of the style loss')
    parser.add_argument('--lr', required=False, default=0.003,
        type=float, help='The learning rate')
    parser.add_argument('--device', required=False, default='cpu',
        type=str, help='Specify the device on which the script is to run')

    options = parser.parse_args()

    if options.device == 'cuda':
        if not torch.cuda.is_available():
            parser.error('[-] Compatible Nvidia GPU not found!')

    return options


def load(cont_path, style_path, device, img_size):
    '''
    A small function for loading and preparing the
    necessities.\n
    `cont_path`: Path/Link to the content image.\n
    `style_path`: Path/Link to the style image.\n
    `device`: The device for the model and the images.\n
    `img_size`: The desired size for the image.
    '''

    content_image = load_image(cont_path, device, img_size)
    _, _, w,h = content_image.shape
    style_image = load_image(style_path, device, (w,h))

    target = content_image.clone().requires_grad_(True).to(device)

    vgg = models.vgg19(pretrained=True).features.eval().to(device)

    content_features = get_features(content_image, vgg, layers)
    style_features = get_features(style_image, vgg, layers)


    style_grams = {
        layer: gram_matrix(style_features[layer]) for layer in style_features
        }

    return content_features, style_grams, target, vgg


def main(content_features, style_grams, target, model, learning_rate, alpha, beta,
    steps, result_path):
    '''
    A function which handle the forward and backward pass as well
    as the optimisation of the generated image.\n
    `content_features`: Dictionary which stores the content image features.\n
    `style_grams`: Dictionary which stores the style image gram matrices.\n
    `target`: Tensor which is to be optimised to get the stylised image.\n
    `model`: Model which is used for the style transfer.\n
    `learning_rate`: The learning rate for the optimizer.\n
    `alpha`: The weight of the content loss.\n
    `beta`: The weight of the style loss.\n
    `result_path:` Path where the image is supposed to be stored.
    '''

    optimizer = optim.Adam([target], lr=learning_rate)

    for step in range(1, steps+1):

        total_loss = content_loss = style_loss = 0

        optimizer.zero_grad()

        target_features = get_features(target, model, layers)

        content_loss = torch.mean((target_features['conv4_2'] - 
            content_features['conv4_2'])**2)

        for layer in style_weights:
            target_feature = target_features[layer]
            _, c, w, h = target_feature.size()

            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]

            layer_style_loss = style_weights[layer] * torch.mean((
                target_gram - style_gram)**2)

            style_loss += layer_style_loss/(c*w*h)

        total_loss = alpha*content_loss + beta*style_loss
        total_loss.backward(retain_graph=True)
        optimizer.step()

        if step % 400 == 0:
            print(f'Step({step}/{steps}) => Loss: {total_loss.item():.4f}')
            save_image(target, result_path)



if __name__ == "__main__":
    arguments = get_arguments()
    content_features, style_grams, target, model = load(arguments.content,
        arguments.style, arguments.device, arguments.size)
    main(content_features, style_grams, target, model, arguments.lr,
        arguments.alpha, arguments.beta, arguments.steps, arguments.result)