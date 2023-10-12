from transformers import T5Tokenizer, T5ForConditionalGeneration
# https://huggingface.co/google/flan-t5-xl#model-details
class FLAN_T5_Wrapper:
    def __init__(self, device='cuda:0'):
        print(T5Tokenizer)
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        device_map= {
            "language_model": 0,
            "language_projection": 0        
        }
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto") #torch_dtype=torch.float16
        self.device = device
    def predict(self, context, question):
        
        input_text = f"{context} {question}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        
        outputs = self.model.generate(input_ids)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()