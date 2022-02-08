from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from model.gen import Generator

# Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
A_image_path = "G:\CycleGAN_DATA\horse2zebra\\testA\\n02381460_8980.jpg"
B_image_path = "G:\CycleGAN_DATA\horse2zebra\\testB\\n02391049_560.jpg"
CHECKPOINT_GEN_A = "genHorse.pth.tar"  # A
CHECKPOINT_GEN_B = "genZebra.pth.tar"  # B

# transform
transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])

# data
A_image = Image.open(A_image_path).convert('RGB')
B_image = Image.open(B_image_path).convert('RGB')

A_image_data = transform(A_image).unsqueeze(0).to(device)
B_image_data = transform(B_image).unsqueeze(0).to(device)

# model
gen_A = Generator()
gen_A.load_state_dict(torch.load(CHECKPOINT_GEN_A))
gen_A.to(device)
gen_B = Generator()
gen_B.load_state_dict(torch.load(CHECKPOINT_GEN_B))
gen_B.to(device)

# predict
fake_A = gen_A(B_image_data) * 0.5 + 0.5
fake_B = gen_B(A_image_data) * 0.5 + 0.5
cycle_A = gen_A(fake_B) * 0.5 + 0.5
cycle_B = gen_B(fake_A) * 0.5 + 0.5

# save
stack_A_image = make_grid([A_image_data[0], fake_B[0], cycle_A[0]])
save_image(stack_A_image, "images/pred_H_Z_H1.png")
stack_B_image = make_grid([B_image_data[0], fake_A[0], cycle_B[0]])
save_image(stack_B_image, "images/pred_Z_H_Z1.png")

