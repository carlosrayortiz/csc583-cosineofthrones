#!/usr/bin/env bash
set -e

###########################################
#  Cosinify - Cloud Run Submit + Deploy
###########################################

SERVICE_NAME="cosinify"
REGION="us-central1"
REPO="cosinify"
IMAGE_NAME="app"
SECRET_NAME="OPENAI_API_KEY"
HF_SECRET="HF_TOKEN"
ENV_FILE=".env"

### NEW ###
GCS_BUCKET="cosinify-artifacts"

# Auto-detect project ID
PROJECT_ID=$(gcloud config get-value project)

if [ -z "$PROJECT_ID" ]; then
    echo "❌ ERROR: No GCP project set. Run this:"
    echo "    gcloud config set project PROJECT_ID"
    exit 1
fi

echo ""
echo "🚀 Deploying Cosinify (API + UI)"
echo "➡️  Project: $PROJECT_ID"
echo "➡️  Region:  $REGION"
echo "➡️  Service: $SERVICE_NAME"
echo ""

###########################################
#  Validate .env file exists
###########################################
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

###########################################
#  Extract keys from .env
###########################################
echo "🔍 Extracting keys from $ENV_FILE..."

OPENAI_API_KEY=$(grep "^OPENAI_API_KEY=" "$ENV_FILE" | cut -d '=' -f2)
HF_TOKEN=$(grep "^HF_TOKEN=" "$ENV_FILE" | cut -d '=' -f2)

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY not found in $ENV_FILE"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "❌ ERROR: HF_TOKEN not found in $ENV_FILE"
    exit 1
fi

echo "✔ Found OPENAI_API_KEY"
echo "✔ Found HF_TOKEN"

###########################################
#  Create or update OpenAI key
###########################################
echo ""
echo "🔐 Syncing OpenAI key to Secret Manager..."

if gcloud secrets describe "$SECRET_NAME" --project "$PROJECT_ID" >/dev/null 2>&1; then
    echo "✔ Secret exists. Adding new version..."
    gcloud secrets versions add "$SECRET_NAME" \
        --data-file=<(echo -n "$OPENAI_API_KEY") \
        --project "$PROJECT_ID"
else
    echo "❌ Secret does not exist. Creating it now..."
    gcloud secrets create "$SECRET_NAME" \
        --data-file=<(echo -n "$OPENAI_API_KEY") \
        --project "$PROJECT_ID"
    echo "✔ Secret created."
fi

###########################################
#  Create or update HF_TOKEN secret
###########################################
echo ""
echo "🔐 Syncing HF_TOKEN to Secret Manager..."

if gcloud secrets describe "$HF_SECRET" --project "$PROJECT_ID" >/dev/null 2>&1; then
    echo "✔ HF_TOKEN secret exists. Adding new version..."
    gcloud secrets versions add "$HF_SECRET" \
        --data-file=<(echo -n "$HF_TOKEN") \
        --project "$PROJECT_ID"
else
    echo "❌ HF_TOKEN secret does not exist. Creating it now..."
    gcloud secrets create "$HF_SECRET" \
        --data-file=<(echo -n "$HF_TOKEN") \
        --project "$PROJECT_ID"
    echo "✔ HF_TOKEN secret created."
fi

###########################################
#  Build & Push Container with Cloud Build
###########################################
echo ""
echo "🔧 Building container image..."

gcloud builds submit \
    --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}

###########################################
#  Deploy to Cloud Run
###########################################
echo ""
echo "🚢 Deploying to Cloud Run with secrets + GCS bucket..."

gcloud run deploy "$SERVICE_NAME" \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME} \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory 8Gi \
    --cpu 4 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 5 \
    --set-secrets OPENAI_API_KEY=${SECRET_NAME}:latest \
    --set-secrets HF_TOKEN=${HF_SECRET}:latest \
    --set-env-vars GCS_BUCKET=$GCS_BUCKET

###########################################
#  Bind IAM for secret access
###########################################
echo ""
echo "🔐 Ensuring Cloud Run service has secret access..."

SERVICE_ACCOUNT=$(gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --format 'value(spec.template.spec.serviceAccountName)')

gcloud secrets add-iam-policy-binding "$SECRET_NAME" \
    --member=serviceAccount:$SERVICE_ACCOUNT \
    --role="roles/secretmanager.secretAccessor" \
    --project "$PROJECT_ID" >/dev/null

gcloud secrets add-iam-policy-binding "$HF_SECRET" \
    --member=serviceAccount:$SERVICE_ACCOUNT \
    --role="roles/secretmanager.secretAccessor" \
    --project "$PROJECT_ID" >/dev/null

echo "✔ Secret access granted to $SERVICE_ACCOUNT"

###########################################
#  Output deployed URLs
###########################################
echo ""
URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --format 'value(status.url)')

echo "🎉 Deployment complete!"
echo "🌐 Base URL:  $URL"
echo "📌 UI:        $URL/ui"
echo "📌 API:       $URL/api/answer?q=Jon+Snow"
echo ""
echo "Done!"