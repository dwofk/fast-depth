import torch
import torchvision
import argparse
import utils

args = utils.parse_command()
model_path = args.model

checkpoint = torch.load(model_path)
# An instance of your model.
start_epoch = checkpoint['epoch']
best_result = checkpoint['best_result']
model = checkpoint['model']
print("=> loaded best model (epoch {})".format(checkpoint['epoch']))

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224).cuda()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

save_path = "./results/traced_fastdepth_model.pt"
traced_script_module.save(save_path)
print("Saved to ", save_path)