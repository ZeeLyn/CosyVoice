from  cosyvoice.cli.cosyvoice import CosyVoice2
import time
import torchaudio
from tqdm import tqdm
from cosyvoice.utils.file_utils import logging
import torch
import hashlib
import os
import random
import librosa

max_val = 0.8

class CosyVoicePlus(CosyVoice2):
    # def __init__(self, model_dir, load_jit=True, load_onnx=False, fp16=True):
    #     super().__init__(model_dir, load_jit=load_jit, load_onnx=load_onnx, fp16=fp16)
    #     self.speech_token_cache={}
    def generate_seed(self):
        seed = random.randint(1, 100000000)
        return seed
    
    
    def postprocess(self,speech, top_db=60, hop_length=220, win_length=440):
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
        speech = torch.concat([speech, torch.zeros(1, int(self.sample_rate * 0.2))], dim=1)
        return speech
    
    
    def inference_clone(self, tts_text, prompt_text, prompt_speech_16k,prompt_wav, stream=False, speed=1.0, text_frontend=True):
        
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        prompt_text = self.frontend.text_normalize(prompt_text, split=False,text_frontend=text_frontend)
        
        prompt_text_token, prompt_text_token_len = self.frontend._extract_text_token(prompt_text)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=self.sample_rate)(prompt_speech_16k)
        speech_feat, speech_feat_len = self.frontend._extract_speech_feat(prompt_speech_resample)
        
        md5=hashlib.md5()
        md5.update(prompt_wav.encode('utf-8'))
        pth= os.path.join("prompt_pth/",'{}.pth'.format(md5.hexdigest()))
        if os.path.exists(pth):
            # print('加载模型')
            speech_token=torch.load(pth).to(device)
            speech_token_len=torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(device)
            
        else:
            speech_token, speech_token_len = self.frontend._extract_speech_token(prompt_speech_16k)
            torch.save(speech_token,pth)
            
        embedding = self.frontend._extract_spk_embedding(prompt_speech_16k)
        
        start_time = time.time()  
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
                
            tts_text_token, tts_text_token_len = self.frontend._extract_text_token(i)
            model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                       'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                       'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                       'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                       'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 'flow_embedding': embedding}
            
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()  