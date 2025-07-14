#!/bin/bash

# Default values for parameters
DATASET="kidney"
KIDNEY_PATH="data/kidney/"
LUNG_PATH="data/lung/"
COVID_PATH="data/covid/"
HPT_TRIALS=20
HPT_METHOD="bayesian"
SAVE_DIR="hpt_results"
INPUT_SIZE=256
PATCH_SIZE=64

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
        --hpt_trials)
            HPT_TRIALS="$2"
            shift
            shift
            ;;
        --hpt_method)
            HPT_METHOD="$2"
            shift
            shift
            ;;
        --save_dir)
            SAVE_DIR="$2"
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
            echo "  --hpt_trials NUM            Number of hyperparameter tuning trials"
            echo "  --hpt_method METHOD         Hyperparameter tuning method (random, grid, bayesian)"
            echo "  --save_dir DIR              Directory to save results"
            echo "  --input_size SIZE           Input image size"
            echo "  --patch_size SIZE           Size of patches for local refinement"
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

# Run the hyperparameter tuning script
echo "Starting hyperparameter tuning with the following configuration:"
echo "Dataset: $DATASET"
echo "Hyperparameter tuning method: $HPT_METHOD"
echo "Number of trials: $HPT_TRIALS"
echo "Input size: $INPUT_SIZE"
echo "Patch size: $PATCH_SIZE"
echo "Save directory: $SAVE_DIR"

python main.py \
    --mode hpt \
    --dataset $DATASET \
    --kidney_path $KIDNEY_PATH \
    --lung_path $LUNG_PATH \
    --covid_path $COVID_PATH \
    --hpt_method $HPT_METHOD \
    --hpt_trials $HPT_TRIALS \
    --input_size $INPUT_SIZE \
    --patch_size $PATCH_SIZE \
    --save_dir $SAVE_DIR \
    $ABLATION_MODE

echo "Hyperparameter tuning complete!"
