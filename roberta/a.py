from transformers import RobertaConfig, AutoModelForTokenClassification
import torch
import os

root_path = os.getcwd()
local_model = "model/epoch19.pth"

model_name = "roberta-base"
device = "cuda"

local_model = os.path.join(root_path, local_model)
config = RobertaConfig.from_pretrained(model_name, num_labels=10)
model = AutoModelForTokenClassification.from_config(config)
model.load_state_dict(torch.load(local_model))
model = model.to(device)

model.eval()

input = torch.randint(low=0, high=10000, size=[4, 510], dtype=torch.int).to(device)
mask = torch.ones([4, 510], dtype=torch.int).to(device)

with torch.amp.autocast(device, dtype=torch.bfloat16):
    preds = model(input, mask).logits
print(preds)
