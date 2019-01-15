"""
** coded on 15/1/19 **
** DEEP LIGHT **
"""

import torch
import vlc
from gtts import gTTS
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import pickle
import time

encoder_path = 'models/encoder-5-3000.pkl'
decoder_path = 'models/decoder-5-3000.pkl'
vocab_path = 'data/vocab.pkl'

# Model parameters
embed_size = 256
hidden_size = 512
num_layers = 1

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def play(string):
    tts = gTTS(text=string, lang='en')
    tts.save("out.mp3")
    player = vlc.MediaPlayer("out.mp3")
    player.audio_set_volume(100)
    player.play()

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def caption(image):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab),num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    # Prepare an image
    image = load_image(image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    sentence = sentence.replace("<start>", "")
    sentence = sentence.replace("<end>", "")
    
    # return caption
    play(sentence)
    time.sleep(5)
    print(sentence)

caption("./png/12.jpg")
