# MasRouter

## Overview

Masrouter is the first multi-agent system(MAS) routing solution, which integrates all components of MAS into a unified routing framework. Masrouter employs collaboration mode determination, role allocation, and LLM routing through a cascaded controller network, progressively constructing a MAS that balances effectiveness and efficiency.

We provide the code of our paper. The algorithm implementation code is in `MAR` folder, and the experimental code is in `Experiments` folder.

## Quick Start

### Add API keys in `template.env` and change its name to `.env`

```python
URL = "" # the URL of OpenAI LLM backend
KEY = "" # the key for API
```
We recommend that this API be able to access multiple LLMs.

### Download Datasets
Download MMLU, HumanEval, GSM8K, MATH, MBPP datasets, and place them in the corresponding folders.

### Run Masrouter on MBPP by running the following scripts

```bash
python experiments/run_mbpp.py
```

The above code verifies the experimental results of the `mbpp` dataset.

