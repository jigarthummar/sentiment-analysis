# Sentiment Analysis

A robust and accurate sentiment analysis microservice for analyzing text data. This repository contains code for a sentiment analysis API using DistilBERT, deployed on Google Cloud Run.

## Features

- Text sentiment classification with high accuracy (96% on test data)
- RESTful API with clear, consistent endpoints
- Confidence scores and probability distribution for each prediction
- Support for both single text and batch analysis
- Containerized deployment for scalability
- Automatic documentation with Swagger UI

## Architecture

- **Model**: Fine-tuned DistilBERT (distilbert-base-uncased)
- **API Framework**: FastAPI
- **Deployment**: Google Cloud Run
- **Model Storage**: Google Cloud Storage
- **Container**: Docker

## Running the Sentiment Analysis Application

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/jigarthummar/sentiment-analysis.git
cd sentiment-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Google Cloud credentials**
   * Download the service account JSON key from Google Cloud
   * Set the environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

4. **Run the application locally**
```bash
python app.py
```
   * The app will be available at http://localhost:8080

### Deployment to Google Cloud

1. **Run the deployment script**
```bash
chmod +x deploy.sh
./deploy.sh
```

2. **Access the deployed service**
   * The script will output the URL of the deployed service
   * Alternatively, access it at: https://sentiment-analysis-api-273171502228.us-central1.run.app/

## API Usage Examples

### Analyze a single text
```bash
curl -X POST "https://sentiment-analysis-api-273171502228.us-central1.run.app/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text":"I absolutely love this product! The quality is amazing."}'
```

### Analyze multiple texts
```bash
curl -X POST "https://sentiment-analysis-api-273171502228.us-central1.run.app/analyze-batch" \
     -H "Content-Type: application/json" \
     -d '{"texts":["Exceeded my expectations! The packaging was beautiful.", "Meh... not the worst, but definitely not worth the price."]}'
```

### Check service health
```bash
curl "https://sentiment-analysis-api-273171502228.us-central1.run.app/health"
```

### Interactive API documentation
Access the Swagger UI at https://sentiment-analysis-api-273171502228.us-central1.run.app/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message and service information |
| `/health` | GET | Service health check |
| `/analyze` | POST | Analyze sentiment of a single text |
| `/analyze-batch` | POST | Analyze sentiment of multiple texts |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

## Response Format

The API returns sentiment analysis results in the following format:

```json
{
  "text": "I absolutely love this product! The quality is amazing.",
  "sentiment": "Positive",
  "confidence": 99.86,
  "probabilities": {
    "negative": 0.14,
    "positive": 99.86
  }
}
```

## Acknowledgments
- Thanks to the Hugging Face team for providing the DistilBERT model.
- Thanks to the FastAPI team for the excellent web framework.
