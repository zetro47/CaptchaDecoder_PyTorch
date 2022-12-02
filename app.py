import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os

idx2char = {0: '-', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9'}
char2idx = {'-': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10}


def transform(image):
    transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform_ops(image)
 

def decode_predictions(text_batch_logits):
    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2) # [T, batch_size]
    text_batch_tokens = text_batch_tokens.cpu().numpy().T # [batch_size, T]

    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = [idx2char[idx] for idx in text_tokens]
        text = "".join(text)
        text_batch_tokens_new.append(text)

    return text_batch_tokens_new


def remove_duplicates(text):
    if len(text) > 1:
        letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx-1]]
    elif len(text) == 1:
        letters = [text[0]]
    else:
        return ""
    return "".join(letters)


def correct_prediction(word):
    parts = word.split("-")
    parts = [remove_duplicates(part) for part in parts]
    corrected_word = "".join(parts)
    
    if(len(corrected_word) == 5):
        freqs = {}
        for element in set(word):
            freqs.update({element: {'val': 0, 'index': None}})
        prev_element = None
        i = 0
        for element in word:
            if(prev_element == element and not(element=='-') ):
                freqs[element]['val'] += 1
                freqs[element]['index'] = i
            prev_element = element
            i = i + 1
        max_freq = 0
        max_element = None
        for element in freqs:
            if(freqs[element]['val'] > max_freq):
                max_freq = freqs[element]['val']
                max_element = element
        index = freqs[max_element]['index']
        word = word[0:index] + "-" + word[index:]
        return(correct_prediction(word))
            
    return corrected_word



from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/crackCaptcha', methods= ['GET'])
def crackCaptcha():
    url = request.headers["url"]
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    image = transform(image)
    outputs = ort_sess.run(None, {'image': image.numpy().reshape(1, 3, 50, 182)})
    return(correct_prediction(decode_predictions(torch.Tensor(outputs[0]))[0]))

if __name__ == '__main__':
    ort_sess = ort.InferenceSession(os.getcwd() + "/zetrocaptchanet.onnx")
    app.run()
