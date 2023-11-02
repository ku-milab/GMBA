# Age-Aware Guidance via Masking-Based Attention in Face Aging
![architecture](./framework.jpg)

This repository provides the official PyTorch implementation of the following paper:
> **Age-Aware Guidance via Masking-Based Attention in Face Aging**<br>
> [Junyeong Maeng](https://scholar.google.co.kr/citations?user=8yuRvWMAAAAJ&hl=ko)<sup>1,\*</sup>, [Kwanseok Oh](https://scholar.google.co.kr/citations?user=EMYHaHUAAAAJ&hl=ko)<sup>1,\*</sup>, and [Heung-Il Suk](https://scholar.google.co.kr/citations?user=dl_oZLwAAAAJ&hl=ko)<sup>1, 2</sup><br/>
> (<sup>1</sup>Department of Artificial Intelligence, Korea University) <br/>
> (<sup>2</sup>Department of Brain and Cognitive Engineering, Korea University) <br/>
> (* indicates equal contribution) <br/> 
> Official Version: https://doi.org/10.1145/3583780.3615183 <br/>
> Published in 32nd ACM International Conference on Information and Knowledge Management (CIKM), At: Birmingham, UK

> **Abstract:** *Face age transformation aims to convert reference images into synthesized images so that they portray the specified target ages. The crux of this task is to change only age-related areas of the given image while maintaining the age-irrelevant areas unchanged. Nevertheless, a common limitation among most existing models is the struggle to generate high-quality aging images that effectively consider both crucial properties. To address this problem, we propose a novel GAN-based face-aging framework that utilizes age-aware Guidance via Masking-Based Attention (GMBA). Specifically, we devise an age-aware guidance module to adjust age-relevant and age-irrelevant attributes within the image seamlessly. By virtue of its capability, it enables the model to produce realistic age-transformed images that certainly preserve the input's identities while delicately imposing age-related properties. Experimental results show that our proposed GMBA outperformed other state-of-the-art methods in terms of identity preservation and accurate age conversion, as well as providing superior visual quality for age-transformed images.*

## Setup

- Python 3.7.10
- CUDA Version 11.0

1. Nvidia driver, CUDA toolkit 11.0, install Anaconda.

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
```

4. Install various necessary packages
```
pip install numpy tqdm
```

## Training

When using Terminal, directly execute the code below after setting the path


```
python main.py --gpu 0 --batch_size 64 --epoch 100
```
## Citation
If used in your research, please cite the following paper:
```
@inproceedings{maeng2023age,
  title={Age-Aware Guidance via Masking-Based Attention in Face Aging},
  author={Maeng, Junyeong and Oh, Kwanseok and Suk, Heung-Il},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={4165--4169},
  year={2023}
}
```

## Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by
the Korea government (MSIT) No. 2019-0-00079 (Artificial Intelligence Graduate School Program(Korea University)) and No. 2022-
0-00959 ((Part 2) Few-Shot Learning of Causal Inference in Vision and Language for Decision Making).
