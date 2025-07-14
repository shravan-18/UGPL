#!/bin/bash

# Default values for parameters
DATASET="kidney"
KIDNEY_PATH="data/kidney/"
LUNG_PATH="data/lung/"
COVID_PATH="data/covid/"
CHECKPOINT=""
INPUT_SIZE=256
PATCH_SIZE=64
NUM_PATCHES=3
BACKBONE="resnet34"
SAVE_DIR="results"
ANALYZE_ERRORS=false

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
        --checkpoint)
            CHECKPOINT="$2"
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
        --save_dir)
            SAVE_DIR="$2"
            shift
            shift
            ;;
        --analyze_errors)
            ANALYZE_ERRORS="--analyze_errors"
            shift
            ;;
        --num_error_samples)
            NUM_ERROR_SAMPLES="--num_error_samples $2"
            shift
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
            echo "  --checkpoint PATH           Path to model checkpoint"
            echo "  --input_size SIZE           Input image size"
            echo "  --patch_size SIZE           Size of patches for local refinement"
            echo "  --num_patches NUM           Number of patches to extract"
            echo "  --backbone MODEL            Backbone architecture (resnet18, resnet34, resnet50)"
            echo "  --save_dir DIR              Directory to save results"
            echo "  --analyze_errors            Analyze and visualize model errors"
            echo "  --num_error_samples NUM     Number of error samples to analyze"
            echo "  --ablation_mode MODE        Ablation mode (global_only, no_uncertainty, fixed_patches)"
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

# Check if checkpoint is provided
if [ -z "$CHECKPOINT" ]; then
    echo "Error: Checkpoint must be provided for evaluation. Use --checkpoint PATH"
    exit 1
fi

# Run the evaluation script
echo "Starting evaluation with the following configuration:"
echo "Dataset: $DATASET"
echo "Checkpoint: $CHECKPOINT"
echo "Input size: $INPUT_SIZE"
echo "Patch size: $PATCH_SIZE"
echo "Number of patches: $NUM_PATCHES"
echo "Backbone: $BACKBONE"
echo "Save directory: $SAVE_DIR"

python main.py \
    --mode evaluate \
    --dataset $DATASET \
    --kidney_path $KIDNEY_PATH \
    --lung_path $LUNG_PATH \
    --covid_path $COVID_PATH \
    --checkpoint $CHECKPOINT \
    --input_size $INPUT_SIZE \
    --patch_size $PATCH_SIZE \
    --num_patches $NUM_PATCHES \
    --backbone $BACKBONE \
    --save_dir $SAVE_DIR \
    $ANALYZE_ERRORS \
    $NUM_ERROR_SAMPLES \
    $ABLATION_MODE

echo "Evaluation complete!"
