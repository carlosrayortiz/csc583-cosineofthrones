#!/bin/bash
set -e

###
# CONFIGURATION — EDIT THESE VALUES
###
PROJECT_ID="depaul-cosinify"
REGION="us-central1"
BUCKET_NAME="cosinify-artifacts"   # Must be globally unique
ARTIFACTS_DIR="./ragthrones/data/artifacts"
SERVICE_ACCOUNT="${PROJECT_ID}-compute@developer.gserviceaccount.com"


echo "========================================="
echo "🔧 Setting project"
echo "========================================="
gcloud config set project $PROJECT_ID


echo "========================================="
echo "📌 Enabling required APIs"
echo "========================================="
gcloud services enable storage.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable iam.googleapis.com


echo "========================================="
echo "🏗 Creating GCS bucket: gs://${BUCKET_NAME}"
echo "========================================="
gsutil mb -p $PROJECT_ID -l $REGION gs://${BUCKET_NAME}/ || \
    echo "Bucket already exists, continuing."


echo "========================================="
echo "🔧 Fetching project number"
echo "========================================="
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
echo "Using Cloud Run/Compute SA: $SERVICE_ACCOUNT"


echo "========================================="
echo "👁 Setting bucket permissions for Cloud Run SA"
echo "========================================="
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectViewer"


echo "========================================="
echo "📤 Uploading artifacts"
echo "    Local dir: $ARTIFACTS_DIR"
echo "    GCS bucket: gs://${BUCKET_NAME}/"
echo "========================================="

if [ ! -d "$ARTIFACTS_DIR" ]; then
    echo "ERROR: Artifacts directory not found: $ARTIFACTS_DIR"
    exit 1
fi

gsutil -m cp -r "${ARTIFACTS_DIR}/*" "gs://${BUCKET_NAME}/"


echo "========================================="
echo "🔍 Verifying uploaded files"
echo "========================================="
gsutil ls -r gs://${BUCKET_NAME}/


echo "========================================="
echo "🎉 DONE"
echo "Your Cloud Run service now has read access to:"
echo "   gs://${BUCKET_NAME}"
echo
echo "Update your app to load artifacts from GCS instead of local disk."
echo "========================================="