#!/bin/bash
set -e

# Configuration
PROJECT_ID="distilledbirt"
REGION="us-central1" # Change to your preferred region
BUCKET_NAME="distilbert-model-6010"
SERVICE_NAME="sentiment-analysis-api"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Authenticate with Google Cloud${NC}"
gcloud auth login
gcloud config set project $PROJECT_ID

echo -e "${YELLOW}Step 2: Verify GCS bucket exists${NC}"
gsutil ls -b gs://$BUCKET_NAME &>/dev/null || { echo "Error: Bucket $BUCKET_NAME does not exist"; exit 1; }

echo -e "${YELLOW}Step 3: Verifying model and tokenizer files in GCS${NC}"
gsutil ls gs://$BUCKET_NAME/distilbert_model/model.pth &>/dev/null || { echo "Error: Model file not found in bucket"; exit 1; }
gsutil ls gs://$BUCKET_NAME/distilbert_model/tokenizer &>/dev/null || { echo "Error: Tokenizer file not found in bucket"; exit 1; }
echo "✓ Model and tokenizer found in GCS bucket"

echo -e "${YELLOW}Step 4: Building and pushing Docker image${NC}"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"
gcloud builds submit --tag $IMAGE_NAME

echo -e "${YELLOW}Step 5: Deploying to Cloud Run${NC}"
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300s \
  --set-env-vars="BUCKET_NAME=$BUCKET_NAME"

echo -e "${GREEN}Deployment completed!${NC}"
gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)'