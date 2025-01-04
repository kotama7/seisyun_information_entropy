# Seisyun Information Entropy

This repository contains the implementation and data for quantifying the concept of "Seisyun" (青春), a term widely used in Japan to describe youth, through a novel measure called **Seisyun Information Entropy**. The method combines advanced natural language processing techniques and large language models (LLMs) to establish a quantitative framework for analyzing and understanding this abstract concept.

## Overview

The study introduces a mathematical and computational framework to measure the "youthfulness" of a text by considering three key factors:

1. **Unusualness**: Using the probability of word predictions from pre-trained BERT models to calculate unexpectedness in text.
2. **Positivity**: Sentiment analysis to evaluate the positive nature of the text.
3. **Fluency**: Evaluating the grammatical correctness of the text.

These metrics are integrated into a single measure, **Seisyun Information Entropy**, using information theory principles.

## Formula

The Seisyun Information Entropy for a given text \(S\) is defined as:

```math
I_{adolescence}(S) = -\log_2 P_{unusual}(S) - \log_2(1 - P_{positive}(S)) - \log_2(1 - P_{fluency}(S))
```

Where:
- $P_{unusual}(S)$: The average word prediction probability.
- $P_{positive}(S)$: The probability that the text is classified as positive.
- $P_{fluency}(S)$: The probability that the text is grammatically correct.



## Methodology

The framework leverages multiple pre-trained BERT models:

1. **Unusualness**:
   - [Tohoku BERT (Japanese)](https://huggingface.co/tohoku-nlp/bert-base-japanese)
   - Predicts word probabilities to assess unexpectedness.

2. **Positivity**:
   - [Sentiment-Enhanced BERT](https://huggingface.co/koheiduck/bert-japanese-finetuned-sentiment)
   - Analyzes sentiment polarity.

3. **Fluency**:
   - [Fluency-Scoring BERT](https://huggingface.co/liwii/fluency-score-classification-ja)
   - Evaluates grammatical correctness.

## Repository Structure

- `module/`: Contains the core modules for calculating Seisyun Information Entropy.
- `main.py`: The main script to execute the entropy calculations.
- `seisyun.csv`: Dataset containing text samples related to "Seisyun".
- `goodness.csv`: Dataset with human-annotated "youthfulness" scores.
- `comparison.csv`: Results comparing calculated entropy with human scores.
- `plot.png`: Visualization of the comparison results.
- `requirement.txt`: List of dependencies required to run the project.
- `.gitignore`: Specifies files and directories to be ignored by git.
- `LICENSE`: License information for the repository.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Hugging Face Transformers

Install dependencies:
```bash
pip install -r requirement.txt
```

### Usage

To calculate Seisyun Information Entropy for a given text, run:
```bash
python main.py --input "Your text here"
```

For batch processing, ensure your input data is formatted appropriately and run:
```bash
python main.py 
```

## Results

Experimental results demonstrated a weak positive correlation (\(r = 0.333\)) between Seisyun Information Entropy and manually generated "youthfulness" rankings of texts. The significance test (\(p = 0.035\)) confirmed the statistical validity of the measure at a 5% significance level.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
