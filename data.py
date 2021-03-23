from torch.utils.data import Dataset
import soundfile as sf
from transformers import Wav2Vec2Tokenizer
import numpy as np


# define Data class from Dataset
class Data(Dataset):
    def __init__(self, path, label):
        self.path = path
        self.label = label
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    def __getitem__(self, index):
        path = self.path[index]
        audio_input, _ = sf.read(path)
        audio_input = self.tokenizer(audio_input, return_tensors="pt").input_values
        audio_input = audio_input.reshape(-1)
        if len(audio_input) < 20000:
            audio_input_  = np.array(list(audio_input) + [0]*(20000-len(audio_input)))
        else:
            audio_input_ = audio_input[:20000]
        return audio_input_, self.label[index]

    def __len__(self):
        return len(self.label)




