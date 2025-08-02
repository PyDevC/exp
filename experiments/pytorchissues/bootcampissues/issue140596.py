import logging
import math
import os

import requests

import torch
from PIL import Image
from torch.nn import functional as F
from transformers import BlipForConditionalGeneration, BlipProcessor

def main():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to("cuda")

    img_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    )
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

    exported = torch.export.export(model, tuple(), inputs, strict=False)

main()
