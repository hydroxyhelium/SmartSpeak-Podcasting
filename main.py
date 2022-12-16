import whisper
from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_RATE = 16000
SPEAKER_CUTOFF = 25


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
for i in range(0,5):
    pre_process.append(preprocess_wav(audio_fragments[i]))

lex_embed = np.array([encoder.embed_utterance(wav) for wav in pre_process]).mean(axis=0)
guest_embed = None 


distict_audio_found = False
index_audio = 0 

for index, wav in enumerate(audio_fragments):
    embed_mapping = encoder.embed_utterance(wav)

    if(np.dot(embed_mapping,lex_embed)<= 0.80):
        print("dot product is ")
        print(np.dot(embed_mapping,lex_embed))
        guest_embed = embed_mapping
        distict_audio_found = True
        index_audio = index
        break 

lex_speech = []
guest_speech = []

for index,wav in enumerate(audio_fragments[0:100]):
    wav_embedding = encoder.embed_utterance(wav)

    if(guest_embed is not None):
        lex_likelyhood = np.dot(wav_embedding, lex_embed)
        guest_likelyhood = np.dot(wav_embedding, guest_embed)
        res = model.transcribe(wav)

        if(lex_likelyhood>guest_likelyhood):
            lex_speech.append(res["text"].strip("."))

        else:
            guest_speech.append(res["text"].strip("."))

print(lex_speech)
print(guest_speech)


        


