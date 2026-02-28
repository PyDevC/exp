import torch

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

opt_model = torch.compile(model, backend="inductor")
opt_model(torch.randn(1,3,64,64))
