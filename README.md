# NAACL_submit
[DEMO](https://e62dbe923d300aa794.gradio.live/)

## Download Link
[LINK](https://drive.google.com/drive/folders/1gy5pGUT2ws1BdnJmVezCQ8-IcOUZ7jBz?usp=drive_link)
- Qualitative Evaluation : Qualitative Evaluation Results of Our Model and Other Models
- Quantitative Evaluation : Quantitative Evaluation Results of Our Model
- xllava.tar.gz : Our Model(X-LLaVA)

## Dataset
- Pretraining Dataset: We use image-caption datasets LLaVA-CC3M and its machine-traslated Korean counterpart, LLaVA-KoCC3M.
- VIF Dataset: We utilize GPT-generated multimodal instruction-following data from both LLaVA-Instruct-158K and KoLLaVA-Instruct-158K. Additionally, we use our dataset, mvif, which is produced by the GPT4-V model and encompasses three languages: English, Korean, and Chinese. This dataset is categorized into four types:
    - (1) Object-centric
    - (2) Location-centric
    - (3) Atmosphere-centric
    - (4) Conversation.


## Usage
```
$ pip install -r requirements.txt
```

```
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import math

import requests
from io import BytesIO


disable_torch_init()
model_path = 'path of downloaded model'
model_base = None
model_name = get_model_name_from_path(model_path)
tokenizer, model, l_image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

vision_tower = model.get_vision_tower()
vision_tower.load_model()
vision_tower.to(device='cuda:0', dtype=torch.float16)

image_processor = vision_tower.image_processor

conv_mode = 'llava_llama_2'

if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in conv_mode:
    conv_mode = conv_mode + '_mmtag'

# image_file_dir = 'your_image_path'
# image = Image.open(image_file_dir).convert('RGB')
    
image_url = 'https://www.state.gov/wp-content/uploads/2023/07/shutterstock_1222044724v2-768x476.jpg'
response = requests.get(image_url)
image_data = BytesIO(response.content)

# PIL을 사용하여 이미지 열기
image = Image.open(image_data).convert('RGB')

qs = '''Please explain the atmosphere you feel when you see the image'''

if model.config.mm_use_im_start_end:
    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
else:
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
image_tensor = process_images([image], image_processor, model.config)[0]

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

input_ids = input_ids.unsqueeze(0).to(device='cuda', non_blocking=True)
image_tensor = image_tensor.unsqueeze(0)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        max_new_tokens=512,
        use_cache=True,
        )

input_token_len = input_ids.shape[1]

outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
outputs = outputs.strip()


print('Outputs')
print(outputs)
```
<figure>
  <img src="./figs/example.jpg" width="500" >
</figure>

**Output   
The image exudes a majestic and otherworldly atmosphere, with the ancient ruins of a pyramid set against a dramatic sky. The sunlight pierces through the clouds, creating a beam of light that illuminates the pyramid, highlighting its intricate details and giving it a sacred, almost ethereal quality. The vibrant colors of the sky, ranging from deep blues to vivid oranges and yellows, suggest a powerful and dynamic natural force at work. The overall mood is one of awe and reverence, as if the viewer is witnessing a moment of great historical significance or a sacred site that has stood the test of time.**




## Hardware and Software  
NVIDIA RTX A6000 
- nvidia driver : 530.30.02
- CUDA version : 11.7


## Evaluation results by GPT4-Vision
- Korean answer preference
<figure>
  <img src="./figs/XLLaVA_vs_Others_KO.png" width="500" >
</figure>

- English answer preference
<figure>
  <img src="./figs/XLLaVA_vs_Others_EN.png" width="500" >
</figure>


## Human Evaluation results
- Korean answer preference
<figure>
  <img src="./figs/XLLaVA_vs_Others_HumanEval_KO.png" width="500" >
</figure>

