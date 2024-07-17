# Learning-to-Brighten-from-Attention

### Results
- Results on LOL-real, LOL-syn, SID and SMID datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1u0RaFEhRBZAQOIOJ9T3Q3czUVMl5-YXT?usp=sharing)

- Results of models trained on the LOL-real dataset on LIME, NPE, MEF, DICM, and VV datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1tCT7U3sIUb-O60xzgqV43J6IjJKZfHA9?usp=sharing)

- Results of models trained on the LOL-syn dataset on LIME, NPE, MEF, DICM, and VV datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1p6CkkTkw0EmwQy2IM6M_Ns_jXGlW4q-U?usp=sharing)


&nbsp;

<details close>
<summary><b>Performance on LOL-real, LOL-syn, SID and SMID:</b></summary>

![results1](./results/1_1.png)
![results1](./results/1_2.png)

</details>


<details close>
<summary><b>Performance of models trained on the LOL-real dataset on LIME, NPE, MEF, DICM, and VV:</b></summary>

![results2](./results/2_1.png)
![results1](./results/2_2.png)

</details>


<details close>
<summary><b>Performance of models trained on the LOL-syn dataset on LIME, NPE, MEF, DICM, and VV:</b></summary>

![results2](./results/3_1.png)
![results1](./results/3_2.png)

</details>

&nbsp;

##  Test

If you want to test the model, just run like this (you can specify your image path)
```
python test.py --device cuda  --testDir your_path  --resultDir ./results/ --ckptDir ./ckpt/LOL-real.pk
```

Enhance results will be saved in */reaults/ if `resultDir` is not specified!

