import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC


class Wav2Vec2(nn.Module):
    def __init__(self, n_classes):
        super(Wav2Vec2, self).__init__()
        self.Wav2Vec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.fn1 = nn.Linear(62*32, 256)  
        self.fn2 = nn.Linear(256, n_classes) 
    
    def forward(self, audio_input):
        batch = audio_input.shape[0]
        output = self.Wav2Vec(audio_input).logits
        output = F.relu(self.fn1(output.reshape(batch,-1)))
        output = self.fn2(output)
        output = F.softmax(output.reshape(batch,-1),1)
        return output


