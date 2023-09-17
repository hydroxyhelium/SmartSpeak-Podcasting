## SmartSpeak-Podasting using Resemblyzer, Whisper AI, GPT-3 

SmartSpeaking-Podcast is a tool that automatically transcribes audio files of podcasts, performs speaker diarization, and generates text based on the conversation between the speakers.

### Prerequisites:-

Before you begin, make sure you have the following tools installed on your machine:

- Python 3.7 or later
- Resemblyzer
- whisper AI
- Cohere API

### Installation

__Before installation move main.py and audio.mp3 into the Resemblyzer folder__

To install the required packages, run the following command:

```python 

pip install -r requirements.txt

```

### Usage

To use SmartSpeaking-Podcast, follow these steps:

1. Place the audio file of the podcast in the same directory as the script.

2. Run the following command:

```
python3 main.py 
```

3. The script will transcribe the audio file, classify the speakers, and generate text based on the conversation between them. The output will be stored in a file called transcription.txt.

### Customization

You can customize the behavior of SmartSpeaking-Podcast by modifying the following parameters in the script:

- `model_name`: The name of the chat-GPT model to use for generating text.
- `max_length` : The maximum length of the generated text in tokens.
- `top_k` : The number of top-k tokens to consider when generating text.
- `top_p`: he probability threshold for the top-p filtering when generating text.

### Credits 

SmartSpeaking-Podcast was developed by Priyanshu Sharma.

Special thanks to the amazing [Daniel Munoz](https://www.linkedin.com/in/munozai/)

