import whisper
from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_RATE = 16000
SPEAKER_CUTOFF = 18


## Aim of this project is to separate out the speech of Lex Fridman and his guest. 
## It is know that he speaks for the first 20 second at least. 

model = whisper.load_model("base")

audio = whisper.load_audio("audio.mp3")
# note here that audio is a numpy array so we can use numpy array operations here 

encoder = VoiceEncoder()

total_duration = len(audio)/SAMPLE_RATE

number_of_breaks = int(total_duration/SPEAKER_CUTOFF)
print(len(audio)/SAMPLE_RATE)

audio_fragments = np.array_split(audio, number_of_breaks) 

pre_process = []
for i in range(10, 19):
    pre_process.append(preprocess_wav(audio_fragments[i]))

test_embeds = np.array([encoder.embed_utterance(wav) for wav in pre_process]).mean(axis=0)

distict_audio_found = False
index_audio = 0 

for index, wav in enumerate(audio_fragments):
    embed_mapping = encoder.embed_utterance(wav)

    if(np.dot(embed_mapping,test_embeds)<= 0.65):
        print("dot product is ")
        print(np.dot(embed_mapping,test_embeds))
        distict_audio_found = True
        index_audio = index
        break 

if distict_audio_found:
    print(index_audio)
    res = model.transcribe(audio_fragments[index_audio])
    print(res["text"])


# result = model.transcribe(audio[0: 30*SAMPLE_RATE])

# print(result["text"])


