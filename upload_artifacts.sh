#!/bin/bash
set -e

### CONFIG ###
PROJECT_ID="depaul-cosinify"
BUCKET_NAME="cosinify-artifacts"
ARTIFACTS_DIR="ragthrones/data/artifacts"

echo "========================================="
echo "📤 Uploading vectorstore artifacts to GCS"
echo "========================================="

if [ ! -d "$ARTIFACTS_DIR" ]; then
    echo "❌ ERROR: Artifacts directory not found at: $ARTIFACTS_DIR"
    exit 1
fi

echo "Uploading:"
ls -lh "$ARTIFACTS_DIR"

echo
echo "→ gsutil cp ${ARTIFACTS_DIR}/* gs://${BUCKET_NAME}/"
gsutil -m cp "${ARTIFACTS_DIR}/*" "gs://${BUCKET_NAME}/"

echo
echo "========================================="
echo "🔍 Verifying contents in GCS"
echo "========================================="

gsutil ls -lh "gs://${BUCKET_NAME}/"

echo
echo "========================================="
echo "🎉 DONE — Artifacts uploaded"
echo "Cloud Run will now download the freshly rebuilt:"
echo "  • df_aug.pkl"
echo "  • bm25.pkl"
echo "  • faiss.index"
echo "========================================="