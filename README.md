<!-- <h1 align="center">🌿Sunsimiao-V(孙思邈): 多模态医疗大模型</h1> -->
<div align=center><img src ="./Sunsimiao-V-logo.jpg"/></div>  

<h3 align="center">慧眼明医路，守护健康途</h3>

<p align="center">
<a href="https://github.com/thomas-yanxin/Sunsimiao-V"><img src="https://img.shields.io/badge/GitHub-24292e" alt="github"></a>
<a href="https://huggingface.co/collections/thomas-yanxin/sunsimiao-v-6641e52e2803b4fe0d88f451"><img src="https://img.shields.io/badge/-HuggingFace-yellow" alt="HuggingFace"></a>
<a href="https://www.modelscope.cn/models/thomas/Sunsimiao-V-Phi3/summary"><img src="https://img.shields.io/badge/ModelScope-blueviolet" alt="modelscope"></a>
<a href="https://wisemodel.cn/models/thomas/Sunsimiao-V-Phi3/intro"><img src="https://img.shields.io/badge/WiseModel-561253" alt="WiseModel"></a>
</p> 

<div align="center">

[![GitHub license](https://img.shields.io/github/license/thomas-yanxin/Sunsimiao-V
)](https://github.com/thomas-yanxin/Sunsimiao-V/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/thomas-yanxin/Sunsimiao-V)](https://github.com/thomas-yanxin/Sunsimiao-V/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/thomas-yanxin/Sunsimiao-V)](https://github.com/thomas-yanxin/Sunsimiao-V/fork)
[![GitHub Contributors](https://img.shields.io/github/contributors/thomas-yanxin/Sunsimiao-V)](https://github.com/thomas-yanxin/Sunsimiao-V/graphs/contributors)  
</div>



## 模型列表
| 模型名称 | 模型参数 | 🤗 HuggingFace 下载 | 🤖 ModelScope 下载 | ✡️ WiseModel 下载 |
| :----: | :----: | :----: | :----: | :----: |
| Sunsimiao-V-Phi3 | 4B | [thomas-yanxin/Sunsimiao-V-Phi3](https://huggingface.co/thomas-yanxin/Sunsimiao-V-Phi3) | [thomas/Sunsimiao-V-Phi3](https://www.modelscope.cn/models/thomas/Sunsimiao-V-Phi3/summary) | [thomas/Sunsimiao-V-Phi3](https://wisemodel.cn/models/thomas/Sunsimiao-V-Phi3/intro) |

## 快速开始

- Chat by pipeline

```Python
from transformers import pipeline
from PIL import Image    
import requests

model_id = "thomas-yanxin/Sunsimiao-V-Phi3"
pipe = pipeline("image-to-text", model=model_id, device=0)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"

image = Image.open(requests.get(url, stream=True).raw)
prompt = "<|user|>\n<image>\nWhat does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud<|end|>\n<|assistant|>\n"

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
```

- Chat by pure transformers
```Python
import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "xtuner/llava-phi-3-mini-hf"

prompt = "<|user|>\n<image>\nWhat are these?<|end|>\n<|assistant|>\n"
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_id)


raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

```

## 🙇‍ ‍致谢
```
@misc{2023xtuner,
    title={XTuner: A Toolkit for Efficiently Fine-tuning LLM},
    author={XTuner Contributors},
    howpublished = {\url{https://github.com/InternLM/xtuner}},
    year={2023}
}
```