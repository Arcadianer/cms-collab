from huggingface_hub import hf_hub_download
from generator import Segment, load_csm_1b
import torchaudio
import time
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
from fastapi.responses import StreamingResponse, Response
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import hashlib
import datetime
import uvicorn
import re
import argparse
import soundfile as sf
from contextlib import asynccontextmanager
from pyngrok import ngrok

model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, "cuda")
app = FastAPI()
context_cache={}

def generate_context(voice,mode="default",speaker=0):
    if voice not in context_cache.keys():
        print(f"[tts] context empty for {voice}. Generating...")
        voice_file=f"voices/{voice}/{mode}.wav"     
        audio_tensor, sample_rate = torchaudio.load(voice_file)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
        )
        torchaudio.save(f"tmp/{voice}.wav", audio_tensor.unsqueeze(0).cpu(), generator.sample_rate)
        model = WhisperModel("turbo", device="cuda", compute_type="float16")
        batched_model = BatchedInferencePipeline(model=model)
        segments, info = batched_model.transcribe(voice_file, batch_size=16,)
        transcript=""
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            transcript=transcript+segment.text
        context_cache[voice]=Segment(speaker=0,text=transcript,audio=audio_tensor)
        return context_cache[voice]
    else:
        return context_cache[voice]
            
        


def generate_speech(input_text, voice, speed=1.0,prompt='', temperature=0.3, top_p=0.8, top_k=20,response_format="wav"):
    params_infer_code = {
        'temperature': temperature,
        'top_P': top_p,
        'top_K': top_k,
    }
    params_refine_text = {
        'prompt': prompt
    }
    torch.manual_seed(voice)
    start_time = time.perf_counter()
    seg=generate_context("nar")
    wavs = generator.generate(
        text=input_text,
        speaker=voice,
        context=[seg],
        max_audio_length_ms=30_000,
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"[tts] Elapsed time: {elapsed_time} seconds")
    md5_hash = hashlib.md5()
    md5_hash.update(f"{input_text}-{voice}-{speed}".encode('utf-8'))
    datename=datetime.datetime.now().strftime('%Y%m%d-%H_%M_%S')
    filename = datename+'-'+md5_hash.hexdigest() + f".{response_format}"
    wav_file_path=f'tmp/{filename}'
    torchaudio.save(wav_file_path, wavs.unsqueeze(0).cpu(), generator.sample_rate)

    return wav_file_path

voice_mapping = {
    "alloy": "0",
    "echo": "1",
    "fable": "2",
    "onyx": "3",
    "nova": "4",
    "shimmer": "5"
}

def replace_non_alphanumeric(text):
    return re.sub(r'[^\w\s]', ' ', text)

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = 'alloy'
    response_format: str = 'wav'
    speed: float = 1.0
    temperature: float = 0.3
    prompt: str = '[oral_2][laugh_0][break_6]'

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    if not request.model or not request.input or not request.voice:
        raise HTTPException(status_code=400, detail="Missing required parameters")
    
    try:
        input_text = request.input
        speed = float(request.speed)
        temperature = request.temperature
        voice = request.voice 
        voice = voice_mapping.get(voice, '0')
        prompt = input_text
        print(f"[tts] prompt: {prompt}, voice: {voice}")
        wavs = generate_speech(input_text, voice, temperature=temperature,prompt=prompt,response_format=request.response_format)

        return FileResponse(wavs, media_type=f'audio/{request.response_format}')
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    print("[tts] model loaded, warming up for tts interference")
    generate_context("nar")
    generate_speech("I am ready to go!","0")
    print("[tts] ready")
    print("[tunnel] Setting up Ngrok Tunnel")
    public_url = ngrok.connect(5001).public_url
    print(f"[tunnel] ngrok url: {public_url}")
    print("[server] ready for connections")
    uvicorn.run(app, host=args.host, port=args.port)