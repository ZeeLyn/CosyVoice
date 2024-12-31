# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os,io
import sys
import argparse
import logging
import time
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/..'.format(ROOT_DIR))
sys.path.append('{}/../third_party/Matcha-TTS'.format(ROOT_DIR))
from api.cosyvoice_plus import CosyVoicePlus
from cosyvoice.utils.file_utils import load_wav,logging
import torchaudio
from pydub import AudioSegment
from faster_whisper import WhisperModel
from cosyvoice.utils.common import set_all_random_seed

logging.getLogger().setLevel(logging.WARN)

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    for i,j in enumerate(model_output):
         torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], 22050)
    return JSONResponse({'ok':True})

@app.post("/inference_clone")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: str = Form(), speed:float=Form()):
    prompt_speech_16k = cosyvoice.postprocess(load_wav(os.path.join(args.prompt_audio_dir,prompt_wav), 16000))
    
    set_all_random_seed(cosyvoice.generate_seed())
    
    model_output = cosyvoice.inference_clone(tts_text, prompt_text, prompt_speech_16k,prompt_wav,stream=False,speed=speed)
    file= str(time.time())
    
    # tensors=[]
    # for i,j in enumerate(model_output):
    #     # print(j)
    #     # f=os.path.join(args.output_dir,file+"-{}.wav".format(i))
    #     # torchaudio.save(f, j['tts_speech'], 22050)
    #     # waveform, sample_rate=torchaudio.load(f)
    #     # audios.append(waveform)
    #     tensors.append(j['tts_speech'])
    # output= torch.cat(tensors,1)
    # torchaudio.save(os.path.join(args.output_dir,file+".wav"),output,22050)
    
    
    file_list=[]
    final_audio = AudioSegment.empty()
    for idx,data in enumerate(model_output):
        f=os.path.join(args.output_dir,file+"-{}.wav".format(idx))
        file_list.append(f)
        torchaudio.save(f, data['tts_speech'], cosyvoice.sample_rate)
        audio = AudioSegment.from_wav(f)
        final_audio+=audio
        audio=None
    
        

    final_audio.export(os.path.join(args.output_dir,file+".wav"), format="wav")
    
    for i in file_list:
        os.remove(i)

    return JSONResponse({'output':file+'.wav'})


@app.get("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))



@app.post("/inference_audio_to_text")
async def inference_instruct(audio: str = Form()):
    segments, info = whisperModel.transcribe(os.path.join(args.prompt_audio_dir,audio), beam_size=5)
    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    result=''
    for segment in segments:
        # print(segment)
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        result+= segment.text
    return JSONResponse({'text':result})
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    
    parser.add_argument('--output_dir',
                    type=str,
                    default='D:\\CosyVoice\\audio_clone_output',
                    help='克隆输出目录')
    
    parser.add_argument('--prompt_audio_dir',
                    type=str,
                    default='D:\\CosyVoice\\',
                    help='prompt audio目录')
    args = parser.parse_args()
    cosyvoice = CosyVoicePlus(args.model_dir,load_jit=True)
    
    model_size = "large-v2"
    # Run on GPU with FP16
    whisperModel = WhisperModel(model_size, device="cuda", compute_type="float16")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
