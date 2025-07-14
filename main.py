import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import modules
from src.data import prepare_datasets
from src.models import UGPL
from src.training import train_model
from src.evaluation import evaluate_model, analyze_error_cases
from src.utils.helpers import EMA, EarlyStopping

# Dummy function to handle 'visualize' mode (since you mentioned visualize.py is not provided)
def run_visualizations(model, test_loader, device, save_dir, save_format, dataset_name):
    print(f"Visualization functionality is not available. Ignoring visualize request.")

def main():
    parser = argparse.ArgumentParser(description='Uncertainty-Guided Progressive Learning for CT Classification')
    
    # Dataset arguments
    parser.add_argument('--kidney_path', type=str, required=True, help='Path to kidney dataset')
    parser.add_argument('--lung_path', type=str, required=True, help='Path to lung dataset')
    parser.add_argument('--covid_path', type=str, required=True, help='Path to COVID dataset')
    parser.add_argument('--dataset', type=str, choices=['kidney', 'lung', 'covid'], required=True, 
                        help='Dataset to use for training')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.15, help='Test split ratio')
    
    # Model arguments
    parser.add_argument('--input_size', type=int, default=256, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=64, help='Size of patches for local refinement')
    parser.add_argument('--num_patches', type=int, default=3, help='Number of patches to extract')
    parser.add_argument('--backbone', type=str, default='resnet34', 
                        choices=['resnet18', 'resnet34', 'resnet50'], help='Backbone architecture')
    
    # Training modes
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'visualize', 'hpt'], default='train',
                        help='Mode of operation (train, evaluate, visualize, hyperparameter tuning)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint for evaluation/visualization')
    
    # Visualization arguments
    parser.add_argument('--vis_save_format', type=str, choices=['png', 'pdf'], default='pdf',
                        help='Format to save matplotlib/seaborn visualizations')
    parser.add_argument('--vis_save_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    
    # Ablation study arguments
    parser.add_argument('--ablation_mode', type=str, default=None,
                        choices=[None, 'global_only', 'no_uncertainty', 'fixed_patches'],
                        help='Run ablation study with specified mode')
    
    # New arguments for our added features
    parser.add_argument('--scheduler', type=str, default='plateau', 
                        choices=['plateau', 'cosine', 'onecycle'],
                        help='Learning rate scheduler (plateau, cosine, onecycle)')
    parser.add_argument('--use_ema', action='store_true', help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate (default: 0.999)')
    parser.add_argument('--early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--es_patience', type=int, default=4, help='Early stopping patience (default: 4)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    # Hyperparameter tuning arguments
    parser.add_argument('--hpt_trials', type=int, default=20, help='Number of hyperparameter tuning trials')
    parser.add_argument('--hpt_method', type=str, default='random', choices=['random', 'grid', 'bayesian'],
                        help='Hyperparameter tuning method')
    
    # Analysis arguments
    parser.add_argument('--analyze_errors', action='store_true', help='Analyze and visualize model errors')
    parser.add_argument('--num_error_samples', type=int, default=5, help='Number of error samples to analyze')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create save directories if needed
    os.makedirs(args.save_dir, exist_ok=True)
    if args.mode == 'visualize':
        os.makedirs(args.vis_save_dir, exist_ok=True)
    
    # Determine number of classes based on dataset
    if args.dataset == 'kidney':
        num_classes = 4  # Normal, Cyst, Tumor, Stone
        dataset_path = args.kidney_path
    elif args.dataset == 'lung':
        num_classes = 3  # Benign, Malignant, Normal
        dataset_path = args.lung_path
    elif args.dataset == 'covid':
        num_classes = 2  # COVID, Non-COVID
        dataset_path = args.covid_path
    
    # Prepare datasets
    print(f"Preparing {args.dataset} dataset...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        dataset_path, args.dataset, args.input_size, 
        val_split=args.val_split, test_split=args.test_split,
        seed=args.seed
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Skip model initialization for hyperparameter tuning
    if args.mode != 'hpt':
        model = UGPL(
            num_classes=num_classes,
            input_size=args.input_size,
            patch_size=args.patch_size,
            num_patches=args.num_patches,
            backbone=args.backbone,
            ablation_mode=args.ablation_mode
        ).to(device)
        
        # Load checkpoint if provided
        if args.checkpoint:
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
    
    # Execute the specified mode
    if args.mode == 'train':
        # Initialize optimizer
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Initialize scheduler based on user choice
        if args.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=10, factor=0.5
            )
        elif args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr / 100
            )
        elif args.scheduler == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=args.lr,
                epochs=args.epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1000
            )
        
        print("Starting training...")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=args.epochs,
            dataset_name=args.dataset,
            save_dir=args.save_dir,
            use_ema=args.use_ema,
            ema_decay=args.ema_decay,
            early_stopping_patience=args.es_patience
        )
        
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError("Checkpoint must be provided for evaluation mode")
        
        print("Evaluating model...")
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            dataset_name=args.dataset,
            save_dir=args.save_dir
        )
        
        # Analyze errors if requested
        if args.analyze_errors:
            print("Analyzing error cases...")
            error_cases = analyze_error_cases(
                model=model,
                test_loader=test_loader,
                device=device,
                dataset_name=args.dataset,
                save_dir=args.save_dir,
                num_samples=args.num_error_samples
            )
        
        print(results)
        
    elif args.mode == 'visualize':
        if not args.checkpoint:
            raise ValueError("Checkpoint must be provided for visualization mode")
        
        print("Generating visualizations...")
        run_visualizations(
            model=model,
            test_loader=test_loader,
            device=device,
            save_dir=args.vis_save_dir,
            save_format=args.vis_save_format,
            dataset_name=args.dataset
        )
        
    elif args.mode == 'hpt':
        print("Running hyperparameter tuning...")
        
        # Import necessary libraries for hyperparameter tuning
        try:
            if args.hpt_method in ['random', 'grid']:
                from ray import tune
                from ray.tune.schedulers import ASHAScheduler
            elif args.hpt_method == 'bayesian':
                # Use Optuna for Bayesian optimization instead of Ray Tune's BayesOpt
                import optuna
                print("Using Optuna for Bayesian optimization (better support for mixed parameter types)")
        except ImportError:
            if args.hpt_method == 'bayesian':
                print("Bayesian optimization requires optuna. Please install it with:")
                print("pip install optuna")
            else:
                print("Hyperparameter tuning requires ray[tune]. Please install it with:")
                print("pip install 'ray[tune]'")
            return
        
        # Create the ray_results directory with absolute path
        ray_results_dir = os.path.abspath(os.path.join(args.vis_save_dir, 'ray_results'))
        os.makedirs(ray_results_dir, exist_ok=True)
        
        # Use file:// URI scheme for storage_path
        storage_path = f"file://{ray_results_dir}"
        
        # Define the parameter space for different search methods
        if args.hpt_method == 'bayesian':
            # For Bayesian optimization with Optuna, we don't need to pre-define parameters
            # They are handled directly in the objective function
            pass
        else:
            # For random and grid search, we can use all parameter types
            param_space = {
                "lr": tune.loguniform(1e-5, 1e-3),
                "weight_decay": tune.loguniform(1e-6, 1e-3),
                "batch_size": tune.choice([8, 16, 32]),
                "scheduler": tune.choice(["plateau", "cosine", "onecycle"]),
                "use_ema": tune.choice([True, False]),
                "ema_decay": tune.uniform(0.99, 0.9999),
                "backbone": tune.choice(["resnet18", "resnet34", "resnet50"]),
                "num_patches": tune.choice([1, 2, 3, 4, 5])
            }
        
        # Define the training function for hyperparameter tuning (only used for Ray Tune)
        def train_with_params(config, checkpoint_dir=None):
            # Create model with the given hyperparameters
            hpt_model = UGPL(
                num_classes=num_classes,
                input_size=args.input_size,
                patch_size=args.patch_size,
                num_patches=config["num_patches"],
                backbone=config["backbone"],
                ablation_mode=args.ablation_mode
            ).to(device)
            
            # Create optimizer
            hpt_optimizer = optim.Adam(
                hpt_model.parameters(),
                lr=config["lr"],
                weight_decay=config["weight_decay"]
            )
            
            # Create scheduler
            if config["scheduler"] == "plateau":
                hpt_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    hpt_optimizer, 'min', patience=10, factor=0.5
                )
            elif config["scheduler"] == "cosine":
                hpt_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    hpt_optimizer, T_max=args.epochs, eta_min=config["lr"] / 100
                )
            elif config["scheduler"] == "onecycle":
                hpt_scheduler = optim.lr_scheduler.OneCycleLR(
                    hpt_optimizer,
                    max_lr=config["lr"],
                    epochs=args.epochs,
                    steps_per_epoch=len(train_loader),
                    pct_start=0.3,
                    div_factor=25,
                    final_div_factor=1000
                )
            
            # Create EMA if needed (handled by train_model function)
            # Create early stopping (handled by train_model function)
            
            # Train for a limited number of epochs to save time during HPT
            hpt_epochs = min(args.epochs, 30)  # Limit to 30 epochs for HPT
            
            # Create data loaders with the batch size from config
            hpt_train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
            hpt_val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
            
            # Train the model
            history = train_model(
                model=hpt_model,
                train_loader=hpt_train_loader,
                val_loader=hpt_val_loader,
                optimizer=hpt_optimizer,
                scheduler=hpt_scheduler,
                device=device,
                epochs=hpt_epochs,
                dataset_name=args.dataset,
                save_dir=args.save_dir,
                use_ema=config["use_ema"],
                ema_decay=config["ema_decay"],
                early_stopping_patience=args.es_patience
            )
        
        # Configure the search algorithm
        if args.hpt_method == 'bayesian':
            # Use Optuna for Bayesian optimization
            def optuna_objective(trial):
                # Define hyperparameters using Optuna's suggest methods
                config = {
                    "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                    "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                    "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
                    "scheduler": trial.suggest_categorical("scheduler", ["plateau", "cosine", "onecycle"]),
                    "use_ema": trial.suggest_categorical("use_ema", [True, False]),
                    "ema_decay": trial.suggest_float("ema_decay", 0.99, 0.9999),
                    "backbone": trial.suggest_categorical("backbone", ["resnet18", "resnet34", "resnet50"]),
                    "num_patches": trial.suggest_int("num_patches", 1, 5)
                }
                
                # Create model with the suggested hyperparameters
                hpt_model = UGPL(
                    num_classes=num_classes,
                    input_size=args.input_size,
                    patch_size=args.patch_size,
                    num_patches=config["num_patches"],
                    backbone=config["backbone"],
                    ablation_mode=args.ablation_mode
                ).to(device)
                
                # Create optimizer
                hpt_optimizer = optim.Adam(
                    hpt_model.parameters(),
                    lr=config["lr"],
                    weight_decay=config["weight_decay"]
                )
                
                # Create scheduler
                if config["scheduler"] == "plateau":
                    hpt_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        hpt_optimizer, 'min', patience=10, factor=0.5
                    )
                elif config["scheduler"] == "cosine":
                    hpt_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        hpt_optimizer, T_max=args.epochs, eta_min=config["lr"] / 100
                    )
                elif config["scheduler"] == "onecycle":
                    hpt_scheduler = optim.lr_scheduler.OneCycleLR(
                        hpt_optimizer,
                        max_lr=config["lr"],
                        epochs=args.epochs,
                        steps_per_epoch=len(train_loader),
                        pct_start=0.3,
                        div_factor=25,
                        final_div_factor=1000
                    )
                
                # Train for a limited number of epochs to save time during HPT
                hpt_epochs = min(args.epochs, 30)  # Limit to 30 epochs for HPT
                
                # Create data loaders with the batch size from config
                hpt_train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
                hpt_val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
                
                # Train the model
                history = train_model(
                    model=hpt_model,
                    train_loader=hpt_train_loader,
                    val_loader=hpt_val_loader,
                    optimizer=hpt_optimizer,
                    scheduler=hpt_scheduler,
                    device=device,
                    epochs=hpt_epochs,
                    dataset_name=args.dataset,
                    save_dir=args.save_dir,
                    use_ema=config["use_ema"],
                    ema_decay=config["ema_decay"],
                    early_stopping_patience=args.es_patience
                )
                
                # Return the best validation accuracy
                return max(history['val_acc']) if history and 'val_acc' in history else 0.0
            
            # Create and run the Optuna study
            study = optuna.create_study(direction='maximize')
            study.optimize(optuna_objective, n_trials=args.hpt_trials)
            
            # Print the best hyperparameters
            best_config = study.best_params
            print("Best hyperparameters found:")
            for param, value in best_config.items():
                print(f"  {param}: {value}")
            print(f"Best validation accuracy: {study.best_value:.4f}")
            
            # Save the best hyperparameters
            with open(os.path.join(args.save_dir, f"{args.dataset}_best_hparams.txt"), "w") as f:
                f.write("Best hyperparameters:\n")
                for param, value in best_config.items():
                    f.write(f"{param}: {value}\n")
                f.write(f"Best validation accuracy: {study.best_value:.4f}\n")
            
            return  # Exit early for Bayesian optimization
            
        elif args.hpt_method == 'random':
            # Configure the scheduler
            scheduler = ASHAScheduler(
                metric="val_acc",
                mode="max",
                max_t=30,  # Maximum number of epochs
                grace_period=5,  # Minimum number of epochs per trial
                reduction_factor=2
            )
            
            # Run the hyperparameter tuning
            analysis = tune.run(
                train_with_params,
                config=param_space,
                num_samples=args.hpt_trials,
                scheduler=scheduler,
                resources_per_trial={"cpu": 2, "gpu": 0.5},
                storage_path=storage_path,
                name=f"hpt_{args.dataset}"
            )
        
        elif args.hpt_method == 'grid':
            # For grid search, we need to discretize the continuous parameters
            grid_param_space = {
                "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                "weight_decay": [1e-6, 1e-5, 1e-4, 1e-3],
                "batch_size": [8, 16, 32],
                "scheduler": ["plateau", "cosine", "onecycle"],
                "use_ema": [True, False],
                "ema_decay": [0.99, 0.995, 0.999, 0.9995],
                "backbone": ["resnet18", "resnet34", "resnet50"],
                "num_patches": [1, 2, 3, 4, 5]
            }
            
            # Run the grid search
            analysis = tune.run(
                train_with_params,
                config=grid_param_space,
                resources_per_trial={"cpu": 2, "gpu": 0.5},
                storage_path=storage_path,
                name=f"hpt_{args.dataset}"
            )
        
        # Print the best hyperparameters (only for Ray Tune methods)
        if args.hpt_method != 'bayesian':
            best_config = analysis.get_best_config(metric="val_acc", mode="max")
            print("Best hyperparameters found:")
            for param, value in best_config.items():
                print(f"  {param}: {value}")
            
            # Save the best hyperparameters
            with open(os.path.join(args.save_dir, f"{args.dataset}_best_hparams.txt"), "w") as f:
                f.write("Best hyperparameters:\n")
                for param, value in best_config.items():
                    f.write(f"{param}: {value}\n")

if __name__ == '__main__':
    main()
