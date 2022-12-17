import whisper
from resemblyzer import preprocess_wav, VoiceEncoder
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import os
import openai
import json


## NOTE:- the dot product between the Resemble AI indicate how similar two wave-fragments are. Ie. the likelyhood they were spoken by the same speaker. 


## OpenAI secret key is stored inside ../api_key, you would need to create your own API key
## and store it. 
api_path = "/api_key"
token = ""

## Openining the file and retriving the key
with open(api_path, 'r') as f:
    for line in f:
        token = line.strip("\n").strip(" ")
        
print(token)

## sample_rate is the frequency at which sampling is done
SAMPLE_RATE = 16000

# this is the initial speaker cutoff, which gives us time long enough to get embedding for both speakers. 
SPEAKER_CUTOFF = 30


## Aim of this project is to separate out the speech of Lex Fridman and his guest. 
## It is know that he speaks for the first 20 second at least. 

## Whisper Base model is used in this project, but this can be changed according to needs.
model = whisper.load_model("base")
# note here that audio is a numpy array so we can use numpy array operations here 
audio = whisper.load_audio("audio.mp3")


encoder = VoiceEncoder()

total_duration = len(audio)/SAMPLE_RATE


number_of_breaks = int(total_duration/SPEAKER_CUTOFF)
print(len(audio)/SAMPLE_RATE)

## we split up audio into fragments of size SPEAKER_CUTOFF (s), as we want to determine which words were spoken by each speaker.
## the smaller the size of fragments, the better we would be to classify each word. 
## However initally, we want to create embeding out of a longer audio clip spoken entirely by both the speakers.  
audio_fragments = np.array_split(audio, number_of_breaks) 

pre_process = []

## Since we know that speaker_1 is speaking for atleast the first few minutes, we calculate normalize, and remove long pauses from the clips.
for i in range(0,5):
    pre_process.append(preprocess_wav(audio_fragments[i]))

## we average out the embdeing and calculate embedding associated with Lex.
lex_embed = np.array([encoder.embed_utterance(wav) for wav in pre_process]).mean(axis=0)
guest_embed = None 


distict_audio_found = False
index_audio = 0 


## we parse thorugh the audio fragments till we find a 30 second audio clip that is distinct enough from the first one
  
for index, wav in enumerate(audio_fragments):
    embed_mapping = encoder.embed_utterance(preprocess_wav(wav))

    if(np.dot(embed_mapping,lex_embed)<= 0.80):
        print("dot product is ")
        print(np.dot(embed_mapping,lex_embed))
        guest_embed = embed_mapping
        distict_audio_found = True
        index_audio = index
        break 

## These lists represent the sentences, spoken by each speaker. 
lex_speech = []
guest_speech = []

## Now that we found a clear embedding associated with each speaker.
## we take in even smaller audio fragments and compare them with the embeddings we obtained earlier to classify which speaker it was spoken by
## we then use whisper AI to find, what was spoken by the speaker and append it to either of the array depending on which embedding it was more similar to 

SPEAKER_CUTOFF = 2.5
number_of_breaks = int(total_duration/SPEAKER_CUTOFF) 
audio_fragments = np.array_split(audio, number_of_breaks)


## this is the overall conversation that is currently going on between the two speakers. 
## This would later be supplied to openAI GPT to cotinue the conversation.  
cur_conversation = []


for index,wav in enumerate(audio_fragments[0:40]):
    wav_embedding = encoder.embed_utterance(preprocess_wav(wav))

    if(guest_embed is not None):
        lex_likelyhood = np.dot(wav_embedding, lex_embed)
        guest_likelyhood = np.dot(wav_embedding, guest_embed)
        res = model.transcribe(wav)

        cur_conversation.append(res["text"].strip("."))

        if(lex_likelyhood>guest_likelyhood):
            lex_speech.append(res["text"].strip("."))

        else:
            guest_speech.append(res["text"].strip("."))

## these were the words spoken by Lex and the Guest. 
lex_speech = "".join(lex_speech)
guest_speech = "".join(guest_speech)

print(lex_speech)
print(guest_speech)

openai.api_key = token

## using OpenAI chat GPT to continue the conversation. 
res = openai.Completion.create(
  model="text-davinci-003",
  prompt="This is a conversation between two speakers on a podcast,\n"+"["+"".join(cur_conversation)+"]"+"\n continue this conversation.",
  max_tokens=500,
  temperature=0.7
)

print(res["choices"][0]["text"])