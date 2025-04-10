# Image Generator with diffusion

Implementation du papier
[1] [Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in neural information processing systems, 33, 6840-6851.](./https://arxiv.org/pdf/2006.11239)


> **DEADLINE CODE : 18 AVRIL** :
 Attention au temps de debuggage : il faut que nos trois parties marchent individuellement dans les test mais ça va nous prendre beaucoup de temps d'assembler les 3 parties, l'assemblement se fera dans le dossier ``scripts/``


**Structure du repo**
```
.
├── preprocessing/
│   ├── dataset.py
│   ├── config.py
│   └── tests/
│   └── readme.md
├── model/
│   ├── unet.py
│   └── tests/
│   └── readme.md
├── diffusion/
│   ├── diffusion.py
│   └── tests/
│   └── readme.md
├── scripts/
│   ├── train.py
│   └── generate.py
│   └── generations/
├── README.md
└── requirements.txt
```

