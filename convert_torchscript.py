import os
import torch
import torchvision
import argparse
import utils
import models

parser = argparse.ArgumentParser(description='FastDepth evaluation')
parser.add_argument('-m', '--model', type=str, required=True, help="Path to model.")
parser.add_argument('--save-gpu', action='store_true')
args = parser.parse_args()

model_path = args.model

model_state_dict, _, _, _ = utils.load_checkpoint(args.model)
model_state_dict = utils.convert_state_dict_from_gpu(model_state_dict)
model = models.MobileNetSkipAdd(output_size=(224, 224), pretrained=True)
if model_state_dict:
    model.load_state_dict(model_state_dict)

if args.save_gpu:
    print("Saving model on GPU")
    model.to(torch.device("cuda:0"))
    example = torch.rand(1, 3, 224, 224).cuda()
else:
    print("Saving model on CPU")
    model.to(torch.device("cpu"))
    example = torch.rand(1, 3, 224, 224)

model.eval()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

model_dir = os.path.join(*model_path.split('/')[:-1])
model_name = model_path.split('/')[-1]
device_ext = "gpu" if args.save_gpu else "cpu"

save_path = os.path.join(model_dir, 'traced_' + model_name[:-4] + "_" + device_ext + ".pt")
traced_script_module.save(save_path)
print("Saved to ", save_path)