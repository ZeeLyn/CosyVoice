from  cosyvoice.cli.cosyvoice import CosyVoice
import time
import torchaudio
from tqdm import tqdm
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph
import re
from functools import partial
import torch
import hashlib
import os
from cosyvoice.utils.common import set_all_random_seed
import random

class CosyVoicePlus(CosyVoice):
    # def __init__(self, model_dir, load_jit=True, load_onnx=False, fp16=True):
    #     super().__init__(model_dir, load_jit=load_jit, load_onnx=load_onnx, fp16=fp16)
    #     self.speech_token_cache={}
    def generate_seed(self):
        seed = random.randint(1, 100000000)
        return seed
    
    def text_normalize(self, text, split=True,token_max_n=40,token_min_n=30,merge_len=20,comma_split=True):
        text = text.strip()
        if contains_chinese(text):
            if self.frontend.use_ttsfrd:
                text = self.frontend.frd.get_frd_extra_info(text, 'input')
            else:
                text = self.frontend.zh_tn_model.normalize(text)
            
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。")
            text = text.replace(",", "，")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r'[，,、]+$', '。', text)
            print(text)
            texts = list(split_paragraph(text, partial(self.frontend.tokenizer.encode, allowed_special=self.frontend.allowed_special), "zh", token_max_n=token_max_n,
                                         token_min_n=token_min_n, merge_len=merge_len, comma_split=comma_split))
        else:
            if self.frontend.use_ttsfrd:
                text = self.frontend.frd.get_frd_extra_info(text, 'input')
            else:
                text = self.frontend.en_tn_model.normalize(text)
            text = spell_out_number(text, self.frontend.inflect_parser)
            texts = list(split_paragraph(text, partial(self.frontend.tokenizer.encode, allowed_special=self.frontend.allowed_special), "en", token_max_n=token_max_n,token_min_n=token_min_n, merge_len=merge_len, comma_split=comma_split))
        if split is False:
            return text
        return texts
    
    def inference_clone(self, tts_text, prompt_text, prompt_speech_16k,prompt_wav, stream=False, speed=1.0):
        set_all_random_seed(self.generate_seed())
        prompt_text = self.text_normalize(prompt_text, split=False)
        prompt_text_token, prompt_text_token_len = self.frontend._extract_text_token(prompt_text)
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        speech_feat, speech_feat_len = self.frontend._extract_speech_feat(prompt_speech_22050)
        
        md5=hashlib.md5()
        md5.update(prompt_wav.encode('utf-8'))
        pth= os.path.join("prompt_pth/",'{}.pth'.format(md5.hexdigest()))
        if os.path.exists(pth):
            # print('加载模型')
            speech_token=torch.load(pth)
            speech_token_len=torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
        else:
            speech_token, speech_token_len = self.frontend._extract_speech_token(prompt_speech_16k)
            torch.save(speech_token,pth)
            
        embedding = self.frontend._extract_spk_embedding(prompt_speech_16k)
        
        start_time = time.time()  
        for i in tqdm(self.text_normalize(tts_text, split=True)):
            tts_text_token, tts_text_token_len = self.frontend._extract_text_token(i)
            model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                       'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                       'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                       'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                       'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 'flow_embedding': embedding}
            # model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {},推理用时:{}'.format(speech_len, (time.time() - start_time) / speech_len,time.time() - start_time))
                yield model_output
                start_time = time.time()  