## CycleGAN

### Cycle Transfer Results

**From apples to oranges!** 🤔

<div align="center">
    <img src="images\pred_A_B_A2.png" height="120" width="360"/><br>
	<img src="images\pred_B_A_B2.png" height="120" width="360"/><br>
</div>

### Details

- use "MSE" instead of "log" to calculate the adversarial loss(follow the original paper);
- set the identity weight to 0.0 when i train the generators("gen_A & B");

- the last layer of generators is "tanh", "sigmoid" may be better!

### Thanks!

- https://zhuanlan.zhihu.com/p/65310116

- https://zhuanlan.zhihu.com/p/45394148
- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- https://github.com/aladdinpersson/Machine-Learning-Collection

### More Result Pictures

<div align="center">
    <img src="images\pred_A_B_A1.png" height="120" width="360"/><br>
    <img src="images\pred_B_A_B1.png" height="120" width="360"/><br>
    <img src="images\pred_H_Z_H1.png" height="120" width="360"/><br>
    <img src="images\pred_Z_H_Z1.png" height="120" width="360"/><br>
</div>

