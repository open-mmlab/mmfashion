Our current virtual try-on module follows [Toward Characteristic-Preserving Image-based Virtual Try-On Network](https://arxiv.org/abs/1807.07688)
and their [pytorch implementation](https://github.com/sergeywong/cp-vton).

We recommend users to read this paper for details.


|       Config File       |     dataset   |                  Loss function                    |
| :---------------------: | :-----------: | :-----------------------------------------------: |
|        cp_vton.py       |      VTON     |                L1 Loss + VGG Loss                 |
