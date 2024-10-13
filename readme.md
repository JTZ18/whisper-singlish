# Fine-Tuning Whisper Model with PyTorch Lightning

## Overview
This repository is designed to experiment with fine-tuning the Whisper model using PyTorch Lightning. We begin with the Whisper-tiny model and aim to fine-tune it on a small subset of 2,500 samples from the IMDA National Speech Corpus. The primary objective is to develop a model capable of accurately transcribing Singaporean English speech into text.

## Strategy
Our strategy involves first assessing whether fine-tuning the Whisper-tiny model on this limited dataset can effectively reduce the word error rate (WER). If we observe a decrease in WER, we can then extrapolate scaling laws to larger models and datasets. Subsequently, we plan to implement multi-GPU training strategies to facilitate the scaling of our approach to accommodate larger models and more extensive datasets.

## Whisper Model Variants

This repository explores both the distilled versions of the Whisper model and the original OpenAI models. Below is a summary of the models and their respective parameters and performance metrics.

### Distilled Whisper Models
| Model               | Parameters (M) | Long-Form WER ↓ |
|---------------------|----------------|------------------|
| distil-large-v3     | 756            | 10.8             |
| distil-large-v2     | 756            | 11.6             |
| distil-medium.en    | 394            | 12.4             |
| distil-small.en     | 166            | 12.8             |

### Original Whisper Models
| Size   | Parameters (M) | English-only Model | Multilingual Model |
|--------|----------------|--------------------|---------------------|
| tiny   | 39             | ✓                  | ✓                   |
| base   | 74             | ✓                  | ✓                   |
| small  | 244            | ✓                  | ✓                   |
| medium | 769            | ✓                  | ✓                   |
| large  | 1550           |                    | ✓                   |
| turbo  | 798            |                    | ✓                   |