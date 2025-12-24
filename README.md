# AI News Summarizer

A comprehensive Turkish news summarization system using fine-tuned transformer models to automatically generate concise summaries of Turkish news articles from multiple sources.

## 📚 Course Project

This project was developed for a course assignment for **Introduction to AI**. It demonstrates practical applications of machine learning, natural language processing in building a real-world Turkish text summarization system.

## 📋 Overview

This project implements an end-to-end pipeline for Turkish news summarization, including:
- **Data Collection & Preprocessing**: Web scraping and data cleaning from Turkish news sources
- **Model Fine-tuning**: Training multiple models for Turkish text summarization
- **Interactive UI**: User-friendly demo interface for live summarization

## 🛠️ Tech Stack

- **Models**: T5, mT5, Gemma
- **Framework**: Hugging Face Transformers
- **Training**: PyTorch with Accelerate
- **Data Processing**: Pandas, Scikit-learn
- **Evaluation**: ROUGE metrics
- **UI**: Tkinter (Python GUI)
- **GPU Support**: CUDA

## 📊 Data Sources

The project uses news articles from Turkish news agencies:
- **Anadolu Agency (AA)** - `anadolu_ajansi_haberler.csv`
- **Turkish Radio and Television (TRT)** - `trt_haberler.csv`

Both datasets contain article text with corresponding summaries scraped from public resources.

## 🔄 Pipeline Overview

### 1. Data Collection
Web scrapers extract URLs from news sitemaps and fetch article content:
- `sitemap_scraper_*.py` - Extract URLs from sitemaps
- `content_scraper_*.py` - Fetch article and summary text

### 2. Data Preprocessing
- **Turkish Text Cleaning**: Removes stopwords, normalizes text
- **Data Validation**: Removes null/empty entries
- **Train/Eval Split**: Creates balanced splits for training and evaluation. (I did this to solve overlearning because our model was pretrained before, but overfitting wasnt related with the number of datas)

### 3. Model Fine-tuning

#### T5 Model (Primary)
- Base model: `Turkish-NLP/t5-efficient-small-MLSUM-TR-fine-tuned`
- Input: Full article text
- Output: Summary
- Training features:
  - Beam search decoding (num_beams=4 is better)
  - Early stopping with patience
  - Multiple epochs on different dataset sizes

#### mT5 Model
- Base model: `csebuetnlp/mT5_multilingual_XLSum`
- Multilingual support (works across languages)
- Optimized for memory efficiency

#### Gemma Model
- Google's instruction-tuned language model
- Adapted for Turkish summarization task but it took a lot of effort to fine tune, i couldn't implement it

### 4. Evaluation
- Generates summaries on evaluation set
- Computes ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)
- Compares base model vs fine-tuned model performance

### 5. Inference
Interactive UI application for real-time summarization:
- Tkinter-based GUI
- Adjustable parameters (max length, beam search, sampling)
- GPU/CPU auto-detection to run model on.

### Prerequisites
```bash
python 3.8+
torch >= 2.0
transformers >= 4.30
datasets
pandas
scikit-learn
```

