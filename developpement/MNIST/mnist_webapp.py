import gradio as gr
from PIL import Image
import requests
import io
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
import torch
from model import MNISTNet

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def recognize_digit(image):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to PIL Image necessary if using the API method
    image = Image.fromarray(image.astype('uint8'))
    # Transform the PIL image
    tensor = transform(image).to(device)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    model = MNISTNet().to(device)
    # Load the model

    model.load_state_dict(torch.load("./weights/mnist_net.pth"))
    
    # Make prediction
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = outputs.max(1)

    return jsonify({"prediction": int(predicted[0])})

if __name__=='__main__':

    gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=True)