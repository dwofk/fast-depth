import torch
import torchvision
import argparse
import utils
import models

parser = argparse.ArgumentParser(description='FastDepth evaluation')
parser.add_argument('-m', '--model', type=str, required=True, help="Path to model.")
args = parser.parse_args()

model_path = args.model


model_state_dict, _, _, _ = utils.load_checkpoint(args.model)
model_state_dict = utils.convert_state_dict_from_gpu(model_state_dict)
model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)
if model_state_dict:
    model.load_state_dict(model_state_dict)
model.to(torch.device("cuda:0"))

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224).cuda()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

save_path = "./results/fastdepth_1.0.0/traced_fastdepth_1.0.0_model.pt"
traced_script_module.save(save_path)
print("Saved to ", save_path)