#!/bin/bash

# Default values for parameters
DATASET="kidney"
KIDNEY_PATH="data/kidney/"
LUNG_PATH="data/lung/"
COVID_PATH="data/covid/"
BATCH_SIZE=16
EPOCHS=100
LR=0.0001
WEIGHT_DECAY=0.0001
INPUT_SIZE=256
PATCH_SIZE=64
NUM_PATCHES=3
BACKBONE="resnet34"
SCHEDULER="plateau"
SAVE_DIR="checkpoints"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    
    case $key in
        --dataset)
            DATASET="$2"
            shift
            shift
            ;;
        --kidney_path)
            KIDNEY_PATH="$2"
            shift
            shift
            ;;
        --lung_path)
            LUNG_PATH="$2"
            shift
            shift
            ;;
        --covid_path)
            COVID_PATH="$2"
            shift
            shift
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift
            shift
            ;;
        --lr)
            LR="$2"
            shift
            shift
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift
            shift
            ;;
        --input_size)
            INPUT_SIZE="$2"
            shift
            shift
            ;;
        --patch_size)
            PATCH_SIZE="$2"
            shift
            shift
            ;;
        --num_patches)
            NUM_PATCHES="$2"
            shift
            shift
            ;;
        --backbone)
            BACKBONE="$2"
            shift
            shift
            ;;
        --scheduler)
            SCHEDULER="$2"
            shift
            shift
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift
            shift
            ;;
        --use_ema)
            USE_EMA="--use_ema"
            shift
            ;;
        --early_stopping)
            EARLY_STOPPING="--early_stopping"
            shift
            ;;
        --ablation_mode)
            ABLATION_MODE="--ablation_mode $2"
            shift
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --dataset DATASET           Dataset to use (kidney, lung, covid)"
            echo "  --kidney_path PATH          Path to kidney dataset"
            echo "  --lung_path PATH            Path to lung dataset" 
            echo "  --covid_path PATH           Path to COVID dataset"
            echo "  --batch_size SIZE           Batch size for training"
            echo "  --epochs EPOCHS             Number of epochs for training"
            echo "  --lr RATE                   Learning rate"
            echo "  --weight_decay DECAY        Weight decay for optimizer"
            echo "  --input_size SIZE           Input image size"
            echo "  --patch_size SIZE           Size of patches for local refinement"
            echo "  --num_patches NUM           Number of patches to extract"
            echo "  --backbone MODEL            Backbone architecture (resnet18, resnet34, resnet50)"
            echo "  --scheduler SCHEDULER       Learning rate scheduler (plateau, cosine, onecycle)"
            echo "  --save_dir DIR              Directory to save checkpoints"
            echo "  --use_ema                   Use Exponential Moving Average"
            echo "  --early_stopping            Use early stopping"
            echo "  --ablation_mode MODE        Run ablation study (global_only, no_uncertainty, fixed_patches)"
            echo "  --help                      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run the training script
echo "Starting training with the following configuration:"
echo "Dataset: $DATASET"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Input size: $INPUT_SIZE"
echo "Patch size: $PATCH_SIZE"
echo "Number of patches: $NUM_PATCHES"
echo "Backbone: $BACKBONE"
echo "Scheduler: $SCHEDULER"
echo "Save directory: $SAVE_DIR"

python main.py \
    --mode train \
    --dataset $DATASET \
    --kidney_path $KIDNEY_PATH \
    --lung_path $LUNG_PATH \
    --covid_path $COVID_PATH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --input_size $INPUT_SIZE \
    --patch_size $PATCH_SIZE \
    --num_patches $NUM_PATCHES \
    --backbone $BACKBONE \
    --scheduler $SCHEDULER \
    --save_dir $SAVE_DIR \
    $USE_EMA \
    $EARLY_STOPPING \
    $ABLATION_MODE

echo "Training complete!"
