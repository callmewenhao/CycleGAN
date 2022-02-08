from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from model.gen import Generator
from model.disc import Discriminator
from dataset import A2BDataset


# Hyper Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
learning_rate = 1e-5
lambda_identity = 0.1
lambda_cycle = 10
num_epochs = 10

# paths
A_dir = "G:\CycleGAN_DATA\horse2zebra\\trainA"  # horse
B_dir = "G:\CycleGAN_DATA\horse2zebra\\trainB"  # zebra
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_A = "genHorse.pth.tar"  # A 马
CHECKPOINT_GEN_B = "genZebra.pth.tar"  # B 斑马
CHECKPOINT_CRITIC_A = "criticA.pth.tar"
CHECKPOINT_CRITIC_B = "criticB.pth.tar"

# transform
transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


def train_fn(disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, l1, mse):
    A_real_val = 0
    A_fake_val = 0
    # loop = tqdm(loader, leave=True)

    for idx, (A_images, B_images) in enumerate(loader):
        A_images = A_images.to(device)
        B_images = B_images.to(device)

        # train discriminator
        # discriminator A loss
        fake_A = gen_A(B_images)  # 生成A
        d_real_A = disc_A(A_images)  # 判别真A
        d_fake_A = disc_A(fake_A.detach())  # 判别生成A
        A_real_val += d_real_A.mean().item()
        A_fake_val += d_fake_A.mean().item()
        loss_d_real_A = mse(d_real_A, torch.ones_like(d_real_A))  # 对于判别器，真的标签是1，所以和全1矩阵做mse, mse是CycleGAN在训练时使用的trick！
        loss_d_fake_A = mse(d_fake_A, torch.zeros_like(d_fake_A))  # 假的标签是0，所以和全0矩阵做mse
        loss_d_A = loss_d_real_A + loss_d_fake_A

        # discriminator B loss
        fake_B = gen_B(A_images)  # 注释同上
        d_real_B = disc_B(B_images)
        d_fake_B = disc_B(fake_B.detach())
        loss_d_real_B = mse(d_real_B, torch.ones_like(d_real_B))
        loss_d_fake_B = mse(d_fake_B, torch.zeros_like(d_fake_B))
        loss_d_B = loss_d_real_B + loss_d_fake_B

        loss_d = (loss_d_A + loss_d_B) / 2

        opt_disc.zero_grad()
        loss_d.backward()
        opt_disc.step()

        # train generator
        # adversarial loss for both generators
        # fake_A = gen_A(B_images)  # 生成A
        # fake_B = gen_B(A_images)
        d_fake_A = disc_A(fake_A)  # 判别假A
        d_fake_B = disc_B(fake_B)
        loss_d_fake_A = mse(d_fake_A, torch.ones_like(d_fake_A))  # 对于生成器，假的标签为1
        loss_d_fake_B = mse(d_fake_B, torch.ones_like(d_fake_B))

        # cycle loss
        cycle_B = gen_B(fake_A)
        cycle_A = gen_A(fake_B)
        loss_cycle_B = l1(B_images, cycle_B)
        loss_cycle_A = l1(A_images, cycle_A)

        # identity loss 确保内容、颜色等特征的一致性
        identity_A = gen_A(A_images)
        identity_B = gen_B(B_images)
        loss_identity_A = l1(A_images, identity_A)
        loss_identity_B = l1(B_images, identity_B)

        # total loss
        loss_g = (
            loss_d_fake_A + loss_d_fake_B  # 对抗loss
            + (loss_cycle_A + loss_cycle_B) * lambda_cycle  # 循环loss
            + (loss_identity_A + loss_identity_B) * lambda_identity  # identity loss
        )

        opt_gen.zero_grad()
        loss_g.backward()
        opt_gen.step()

        if idx % 200 == 0 and idx > 0:
            save_image(fake_A * 0.5 + 0.5, f"images/A_{idx}.png")
            save_image(fake_B * 0.5 + 0.5, f"images/B_{idx}.png")
            print(f"disc_fakeA:{A_fake_val}, disc_realA:{A_real_val}")


def main():
    disc_A = Discriminator().to(device)
    disc_B = Discriminator().to(device)
    gen_A = Generator().to(device)
    gen_B = Generator().to(device)

    # optim
    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=learning_rate,
        betas=(0.5, 0.999),  # 用于计算梯度以及梯度平方的运行平均值的系数，默认：(0.9, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    # loss func
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    # dataset
    train_dataset = A2BDataset(A_dir, B_dir, transform)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        train_fn(disc_A, disc_B, gen_A, gen_B, train_dataloader, opt_disc, opt_gen, L1, MSE)

        if SAVE_MODEL:
            torch.save(
                gen_A.state_dict(),
                CHECKPOINT_GEN_A,
            )
            torch.save(
                gen_B.state_dict(),
                CHECKPOINT_GEN_B,
            )


if __name__ == "__main__":
    main()
