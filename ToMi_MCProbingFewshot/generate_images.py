from PIL import Image, ImageDraw, ImageFont
import torch
import json
import tqdm
import pandas as pd 
test_df=pd.read_csv('test_stories_df.csv')

latents = torch.load('/afs/cs.pitt.edu/usr0/arr159/Clever_Hans_or_N-ToM/ToMi_MCProbingFewshot/latent_seed.pth')
save_to_dir='/archive2/arr159/tomi_image_generations'
image_path = "/afs/cs.pitt.edu/usr0/arr159/Clever_Hans_or_N-ToM/ToMi_MCProbingFewshot/blank.png"
blank_image = Image.open(image_path)
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

for _, row in tqdm.tqdm(test_df.iterrows()):
    story_id = row['story_ID']
    for item in row['context'].split('\n'):
        story_line = ' '.join(item.split(' ')[1:]) # drop number
        line_num = item.split(' ')[0]
        prompt = f"in cartoon style: {story_line}"
        with torch.autocast("cuda"):
            images = pipe(
                [prompt] * 1,
                guidance_scale=7.5,
                latents = latents,
            )["images"]
        images[0].save(f'{save_to_dir}/{story_id}_line_{line_num}.png')
    
    written_story_image=blank_image.resize((512,512))
    # Create a drawing object
    draw = ImageDraw.Draw(written_story_image)
    
    # Choose a font and size
    font = ImageFont.truetype("/afs/cs.pitt.edu/usr0/arr159/Arial.ttf", 15)  # You may need to specify the path to your font file
    
    # Choose text color
    text_color = (0, 0, 255)  # White
    
    # Choose the position to add text
    text_position = (0, 0)
    
    # Your text content
    # Add text to the image
    draw.multiline_text(text_position, row['context'], font=font, fill=text_color)
    
    
    written_story_image.save(f'{save_to_dir}/{story_id}_written.png')
