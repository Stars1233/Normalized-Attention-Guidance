# Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ChenDY/NAG_FLUX.1-schnell)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://chendaryen.github.io/NAG.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.21179-b31b1b.svg)](https://arxiv.org/abs/2505.21179)
[![Page Views Count](https://badges.toozhao.com/badges/01JWNDV5JQ2XT69RCZ5KQBCY0E/blue.svg)](https://badges.toozhao.com/stats/01JWNDV5JQ2XT69RCZ5KQBCY0E "Get your own page views count badge on badges.toozhao.com")


![](./assets/banner.jpg)
Negative prompting on 4-step Flux-Schnell:
CFG fails in few-step models. NAG restores effective negative prompting, enabling direct suppression of visual, semantic, and stylistic attributes, such as ``glasses``, ``tiger``, ``realistic``, or ``blurry``. This enhances controllability and expands creative freedom across composition, style, and quality—including prompt-based debiasing.


## News

**2025-06-01:** 🤗 Demo for [Flux-Schnell](https://huggingface.co/spaces/ChenDY/NAG_FLUX.1-schnell) and [Flux-Dev](https://huggingface.co/spaces/ChenDY/NAG_FLUX.1-dev) are now available!


## Approach

The prevailing approach to diffusion model control, Classifier-Free Guidance (CFG), enables negative guidance by extrapolating between positive and negative conditional outputs at each denoising step. However, in few-step regimes, CFG's assumption of consistent structure between diffusion branches breaks down, as these branches diverge dramatically at early steps. This divergence causes severe artifacts rather than controlled guidance.

Normalized Attention Guidance (NAG) operates in attention space by extrapolating positive and negative features Z<sup>+</sup> and Z<sup>-</sup>, followed by L1-based normalization and α-blending. This constrains feature deviation, suppresses out-of-manifold drift, and achieves stable, controllable guidance.

![](./assets/architecture.jpg)

## Usage

You can try NAG in `flux_nag_demo.ipynb`, or 🤗 Hugging Face Demo for [Flux-Schell](https://huggingface.co/spaces/ChenDY/NAG_FLUX.1-schnell) and [Flux-Dev](https://huggingface.co/spaces/ChenDY/NAG_FLUX.1-dev)!

Loading Custom Pipeline:

```python
import torch
from src.pipeline_flux_nag import NAGFluxPipeline
from src.transformer_flux import NAGFluxTransformer2DModel


transformer = NAGFluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    token="hf_token"
)
pipe = NAGFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    token="hf_token",
)
pipe.to("cuda")
```

Sampling with NAG:

```python
prompt = "Portrait of AI researcher."
nag_negative_prompt = "Glasses."
# prompt = "A baby phoenix made of fire and flames is born from the smoking ashes."
# nag_negative_prompt = "Low resolution, blurry, lack of details, illustration, cartoon, painting."

image = pipe(
    prompt,
    nag_negative_prompt=nag_negative_prompt,
    guidance_scale=0.0,
    nag_scale=5.0,
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
```


## Citation 

If you find NAG is useful or relevant to your research, please kindly cite our work:

```bib
@article{chen2025normalizedattentionguidanceuniversal,
    title={Normalized Attention Guidance: Universal Negative Guidance for Diffusion Model}, 
    author={Dar-Yen Chen and Hmrishav Bandyopadhyay and Kai Zou and Yi-Zhe Song},
    journal={arXiv preprint arxiv:2505.21179},
    year={2025}
}
```

