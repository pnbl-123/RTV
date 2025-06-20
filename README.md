# Real-Time Virtual Try-On

[![arXiv](https://img.shields.io/badge/arXiv-2506.12348-b31b1b.svg)](https://arxiv.org/abs/2506.12348)

This project supports real-time virtual try-on without the need for any special sensors. All you need is a webcam and a PC with a GPU equivalent to an RTX 3060 or higher.

Unlike other image-based virtual try-on methods, our approach requires training a dedicated network for each garment in a low-barrier, accessible way (see [this paper](https://arxiv.org/abs/2506.10468)).
Our method simply overlay the synthesized garment on the top of human body without removing the original garment to achieve real-time performace.

![Demo GIF](assets/output.gif)

[Watch on YouTube](https://www.youtube.com/watch?v=7hm1yBsFzHc)


## Installation

```
git clone git@github.com:ZaiqiangWu/RTV.git
cd RTV
```

### Environment
```
conda create -n rtv python=3.9
conda activate rtv
pip install -r requirements.txt  
```

### Weights

```
git lfs install
git clone 
```
## BibTeX
```text
@misc{wu2025realtime,
    title={Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments},
    author={Zaiqiang Wu and I-Chao Shen and Takeo Igarashi},
    year={2025},
    eprint={2506.12348},
    archivePrefix={arXiv},
    primaryClass={cs.GR}
}
```

## Contact Us
**Zaiqiang Wu**: [wuzaiqiang@zju.edu.cn](mailto:wuzaiqiang@zju.edu.cn) 