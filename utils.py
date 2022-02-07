import os



def len_images(A_dir, B_dir):
    A_len = len(os.listdir(A_dir))
    B_len = len(os.listdir(B_dir))
    print(f"A len is {A_len}, B len is {B_len}")


def main():
    A_dir = "G:\CycleGAN_DATA\\apple2orange\\trainA"  # 995
    B_dir = "G:\CycleGAN_DATA\\apple2orange\\trainB"  # 1019
    len_images(A_dir, B_dir)


if __name__ == "__main__":
    main()

