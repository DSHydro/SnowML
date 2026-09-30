#!/bin/bash
# Sync Exp1B checkpoint files from SageMaker to S3
# Run this script ON the SageMaker notebook instance terminal

set -e

echo "========================================="
echo "Sync Exp1B Checkpoints to S3"
echo "========================================="
echo ""

# Configuration
S3_BUCKET="s3://uw-echoe/simran/exp1b_checkpoints"
CHECKPOINT_DIR="$HOME/SageMaker"  # Or wherever your checkpoints are saved
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🔍 Looking for checkpoint files..."
echo ""

# Find all .pt checkpoint files
CHECKPOINT_FILES=$(find $CHECKPOINT_DIR -name "checkpoint_*.pt" -type f 2>/dev/null || true)

if [ -z "$CHECKPOINT_FILES" ]; then
    echo "❌ ERROR: No checkpoint files found in $CHECKPOINT_DIR"
    echo ""
    echo "Searching in common locations..."
    find $HOME -name "checkpoint_*.pt" -type f 2>/dev/null | head -20
    echo ""
    echo "Please update CHECKPOINT_DIR in this script to the correct location"
    exit 1
fi

echo "✅ Found checkpoint files:"
echo "$CHECKPOINT_FILES" | while read file; do
    echo "  - $(basename $file) ($(du -h "$file" | cut -f1))"
done
echo ""

# Count files
NUM_FILES=$(echo "$CHECKPOINT_FILES" | wc -l | xargs)
echo "Total: $NUM_FILES checkpoint files"
echo ""

# Calculate total size
TOTAL_SIZE=$(echo "$CHECKPOINT_FILES" | xargs du -ch | tail -1 | cut -f1)
echo "Total size: $TOTAL_SIZE"
echo ""

# Confirm upload
echo "📤 Will upload to: $S3_BUCKET/$TIMESTAMP/"
echo ""
read -p "Proceed with S3 sync? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "========================================="
echo "Uploading to S3"
echo "========================================="
echo ""

# Create S3 destination with timestamp
S3_DEST="$S3_BUCKET/$TIMESTAMP"

# Upload each checkpoint file
echo "$CHECKPOINT_FILES" | while read file; do
    filename=$(basename "$file")
    echo "📤 Uploading: $filename"

    aws s3 cp "$file" "$S3_DEST/$filename" \
        --storage-class STANDARD_IA \
        --metadata "timestamp=$TIMESTAMP,experiment=exp1b"

    if [ $? -eq 0 ]; then
        echo "   ✅ $filename uploaded"
    else
        echo "   ❌ Failed to upload $filename"
    fi
done

echo ""
echo "========================================="
echo "Upload Additional Files"
echo "========================================="
echo ""

# Upload training log if it exists
if [ -f "$HOME/SageMaker/training.log" ]; then
    echo "📤 Uploading training.log..."
    aws s3 cp "$HOME/SageMaker/training.log" "$S3_DEST/training_exp1b.log"
fi

# Upload MLflow database if it exists
if [ -f "$HOME/SageMaker/mlflow.db" ]; then
    echo "📤 Uploading mlflow.db..."
    aws s3 cp "$HOME/SageMaker/mlflow.db" "$S3_DEST/mlflow_exp1b.db"
fi

# Upload any JSON result files
if ls $HOME/SageMaker/*.json 1> /dev/null 2>&1; then
    echo "📤 Uploading result JSON files..."
    aws s3 cp "$HOME/SageMaker/" "$S3_DEST/" \
        --recursive \
        --exclude "*" \
        --include "*.json"
fi

echo ""
echo "✅ Upload complete!"
echo ""
echo "========================================="
echo "Verification"
echo "========================================="
echo ""

# List uploaded files
echo "📋 Files in S3:"
aws s3 ls "$S3_DEST/" --recursive --human-readable

echo ""
echo "========================================="
echo "Summary"
echo "========================================="
echo ""
echo "✅ Checkpoints uploaded to S3"
echo "   Location: $S3_DEST/"
echo ""
echo "Next steps:"
echo "  1. Run download_checkpoints_from_s3.sh on your local machine"
echo "  2. Keep SageMaker instance running if you need to continue work"
echo "  3. Or stop the instance to save costs"
echo ""
echo "To download on your laptop:"
echo "  ./download_checkpoints_from_s3.sh"
echo ""
