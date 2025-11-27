#!/bin/bash
# Batch processing script using wsi_processing_multi_tissue.py

GPU_ID=$1
if [ -z "$GPU_ID" ]; then
    echo "Usage: $0 <gpu_id>"
    exit 1
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate environment
source /data1/houfei/anaconda3/etc/profile.d/conda.sh
conda activate tiatoolbox-dev

# Confirm settings
echo "=========================================="
echo "GPU $GPU_ID Configuration"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using wsi_processing_multi_tissue.py with multi-tissue region support"
python -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPU(s)')"
echo "=========================================="

# Read file list
FILES_LIST="files_gpu_${GPU_ID}.txt"
if [ ! -f "$FILES_LIST" ]; then
    echo "Error: $FILES_LIST not found!"
    exit 1
fi

BASE_DIR="/data"
TOTAL=$(wc -l < "$FILES_LIST")
CURRENT=0

# Process each file
while IFS= read -r FILE; do
    CURRENT=$((CURRENT + 1))
    
    # Handle relative/absolute paths
    if [[ "$FILE" == /* ]]; then
        FULL_PATH="$FILE"
    else
        FULL_PATH="$BASE_DIR/$FILE"
    fi
    
    BASENAME=$(basename "$FULL_PATH" .svs)
    
    # Get file size for display
    FILE_SIZE=$(du -h "$FULL_PATH" | cut -f1)
    
    echo ""
    echo "[$CURRENT/$TOTAL] Processing: $BASENAME on GPU $GPU_ID"
    echo "File size: $FILE_SIZE"
    echo "Time: $(date '+%H:%M:%S')"
    
    # Use default parameters from wsi_processing_multi_tissue.py
    # Default: tile_size=5000, batch_size=512, no normalization
    python wsi_processing_multi_tissue.py \
        --input-dir "/data/tiatoolbox/data" \
        --output "/results/tiatoolbox/results" \
        --single-wsi "$FULL_PATH" \
        > "/results/log_${BASENAME}_gpu${GPU_ID}.txt" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Success: $BASENAME"
        # Check if multiple tissue regions were processed
        if grep -q "Found [2-9]\|[0-9][0-9]* tissue regions" "/results/log_${BASENAME}_gpu${GPU_ID}.txt"; then
            REGIONS=$(grep -o "Found [0-9]* tissue regions" "/results/log_${BASENAME}_gpu${GPU_ID}.txt" | head -1)
            echo "  Note: $REGIONS"
        fi
    else
        echo "✗ Failed: $BASENAME (check log)"
        # Show last error lines
        tail -n 5 "/results/log_${BASENAME}_gpu${GPU_ID}.txt"
    fi
done < "$FILES_LIST"

echo ""
echo "GPU $GPU_ID processing complete!"
echo "Total processed: $CURRENT files"
