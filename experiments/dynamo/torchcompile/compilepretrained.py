import torch
from transformers import BertTokenizer, BertModel

device = "cuda"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model_compile = torch.compile(model).to(device)
text = "Replace me by any text"
encoded_input = tokenizer(text, return_tensors="pt").to(device)
output = model_compile(**encoded_input)
print(output)
