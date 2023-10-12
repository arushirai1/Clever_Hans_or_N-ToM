# https://huggingface.co/Salesforce/blip2-flan-t5-xl
# https://huggingface.co/Salesforce/blip2-flan-t5-xl-coco
# https://huggingface.co/google/flan-t5-xl#model-details
# pip install accelerate
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
import torch
from accelerate import init_empty_weights, infer_auto_device_map

class BLIPv2_FLANT5_Wrapper:
    def __init__(self, device='cuda:0'):
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        # config = Blip2Config.from_pretrained("Salesforce/blip2-flan-t5-xl")

        # max_memory={i: "10GiB" for i in range(8)}
        # with init_empty_weights():
        #     model = Blip2ForConditionalGeneration(config)
        #     device_map = infer_auto_device_map(model, no_split_module_classes=["T5Block"], dtype=torch.float16, max_memory=max_memory)
        # device_map['language_model.lm_head'] = device_map["language_model.encoder.embed_tokens"]
        device_map= {
            "query_tokens": 0,
            "vision_model": 1,
            "language_model": 2,
            "language_projection": 2,
            "qformer": 3,
        }
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map=device_map)
        self.device_map = device_map
        
        self.random_image = Image.open('blank.png').convert('RGB')
        self.device = device
    def predict(self, context, question):
        
        input_text = f"{context} {question}"
        inputs = self.processor(self.random_image, input_text, return_tensors="pt").to(self.device)
        
        out = self.model.generate(**inputs)

        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer.strip()