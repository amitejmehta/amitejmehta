# Transformer-Based Models Implementations

Welcome to this project where I dive deep into the architecture of transformer-based models through a series of Jupyter notebooks with implementations in PyTorch. In order to completely understand transformer architectures, I spent a lot of time trying to learn the fundamentals of deep learning, pytorch, and attention from a lot of different resources. Eventually, I just started trying to code them. And I must say it is perhaps the most underrated tool for understanding. Turns out Andrej Karpathy discovered the same thing for himself many years ago (guess I should have just read his blog first). Whether you are just visiting, eager to learn, or a recruiter/hiring manager I've shared this project with, I encourage you to check back weekly for new implementations.

## What You Will Find

Every notebook is a standalone implementation of a single transformer-based architecture containing sections dedicated to each of the novel components of a given model's architecture. For newer models, each notebook assumes understanding of previously implemented components (ie for llama 2 we assume an understanding of basic multi-head attention when covering multi-query attention). Thus, I recommend going through the notebooks in order. And if you do want to skip, just check earlier notebooks to find discussions of the individual components you have questions about. 

### 1. Basic Attention Mechanisms

- **Notebook**: `01_Attention-Mechanisms.ipynb`
- **Overview**: Self attention, multi-head attention, multi-head attention with caching, multi-query attention (Friday Feb 9), grouped query attention (Friday Feb 9).

### 2. The Original Transformer Model

- **Notebook**: `02_Original_Transformer.ipynb`
- **Overview**: The encoder-decoder structure of the original transformer from Attention is All You Need.

### 4. LLaMA 2 (coming Fri Feb 16)

- **Notebook**: `XX_LLaMA_2.ipynb`
- **Overview**: LLaMA 2 and its breakthrough components, such as RMSNorm, Rotary Positional Embeddings, MQA. Unlike the original code, my implementation does not rely on `fairscale` to optimize the model for distributed workloads.

## Getting Started

1. **Prerequisites**: Python, Deep Learning Fundamentals, PyTorch fundamentals. I recommend this free course https://www.udacity.com/course/deep-learning-pytorch--ud188 (at least lessons 1 through 4) to learn deep learning and pytorch fundamentals. 