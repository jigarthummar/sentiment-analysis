# Sentiment Analysis

A robust and accurate sentiment analysis tool for analyzing text data from various sources. This repository contains code for training models, preprocessing text, and deploying sentiment analysis as an API.

## Features

- Text preprocessing pipeline optimized for sentiment analysis
- Multiple model implementations (BERT, RoBERTa, and traditional ML approaches)
- REST API for easy integration with other applications
- Comprehensive evaluation metrics and visualizations
- Support for analyzing data from multiple sources (Twitter, Reddit, custom text)

## Installation

```bash
# Clone the repository
git clone https://github.com/jigarthummar/sentiment-analysis.git
cd sentiment-analysis

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from sentiment_analyzer import SentimentAnalyzer

# Initialize the analyzer with the default model
analyzer = SentimentAnalyzer()

# Analyze a single text
result = analyzer.analyze("I absolutely love this product! It's amazing.")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2f}")

# Batch analysis
texts = [
    "The customer service was terrible.",
    "It works exactly as described.",
    "I'm not sure if I would recommend this."
]
results = analyzer.analyze_batch(texts)
```

## Models

The repository includes several pre-trained models:

- **BERT-base-sentiment**: Fine-tuned BERT model (recommended for most applications)
- **RoBERTa-sentiment**: Fine-tuned RoBERTa model (highest accuracy but slower)
- **FastText-sentiment**: Lightweight model for resource-constrained environments

To specify which model to use:

```python
analyzer = SentimentAnalyzer(model="roberta-sentiment")
```

## API Usage

Start the API server:

```bash
python -m sentiment_analyzer.api --port 8000
```

Then make requests:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This is an excellent implementation of sentiment analysis!"}'
```

Or using Python:

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/analyze",
    headers={"Content-Type": "application/json"},
    data=json.dumps({"text": "This is an excellent implementation of sentiment analysis!"})
)

print(response.json())
```

## Training Your Own Models

To train on your custom dataset:

```bash
python -m sentiment_analyzer.train \
  --data-path path/to/your/data.csv \
  --model-type bert \
  --output-dir models/custom-model
```

## Evaluation

The repository includes tools for model evaluation:

```bash
python -m sentiment_analyzer.evaluate \
  --model-path models/bert-base-sentiment \
  --test-data path/to/test_data.csv
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@software{sentiment_analysis,
  author = {Jigar Thummar},
  title = {Sentiment Analysis},
  year = {2025},
  url = {https://github.com/jigarthummar/sentiment-analysis}
}
```

## Acknowledgments

- This project was inspired by state-of-the-art sentiment analysis research
- Thanks to the open-source community for valuable resources and tools
