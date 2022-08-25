# iwpodnet_pytorch
Pytorch implementation of IWPOD-Net (https://github.com/claudiojung/iwpod-net).

## What is this
- <b>iwpod.py</b> contains pytoch implementation of IWPOD-NET
- <b>keras_to_torch.py</b> contains function that allows loading original keras wieghts into the pytorch model
- <b>save_keras_weigths.py</b> small script serialize and dump keras weights

## Important
When using remember that PyTorch uses (B,C,W,H) whereas Keras uses (B,W,H,C) tensor convention.