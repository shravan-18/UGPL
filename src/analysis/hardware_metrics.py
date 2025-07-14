import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ptflops import get_model_complexity_info
from thop import profile

from src.models import UGPL, GlobalUncertaintyEstimator, LocalRefinementNetwork

def get_memory_usage_mb():
    """Returns peak GPU memory usage in MB"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_used = torch.cuda.max_memory_allocated() / 1024**2
    return mem_used

def count_parameters(model):
    """Count trainable parameters in millions"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def get_flops(model, input_shape=(1, 1, 256, 256), device='cpu'):
    """Calculate GFLOPs and Params for a model on the given device"""
    model = model.to(device)
    inputs = torch.randn(input_shape).to(device)
    macs, params = profile(model, inputs=(inputs,), verbose=False)
    return macs / 1e9, params / 1e6  # GFLOPs, Params (in Millions)

def measure_inference_time(model, device='cuda', input_shape=(1, 1, 256, 256), 
                          runs=100, warmup=10, return_std=False):
    """Measure average inference time in milliseconds with warmup"""
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            torch.cuda.synchronize()
    
    # Timed runs
    timings = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # Convert to ms
    
    if return_std:
        return np.mean(timings), np.std(timings)
    return np.mean(timings)

def measure_full_pipeline_metrics(model_configs, input_sizes=(256, 224, 192), device='cuda'):
    """
    Measure efficiency metrics for different model configurations and input sizes
    
    Args:
        model_configs: List of model configurations (name, num_classes, etc.)
        input_sizes: List of input image sizes to test
        device: Device to run on
    
    Returns:
        DataFrame with all metrics
    """
    results = []
    
    for config in model_configs:
        model_name = config['name']
        num_classes = config['num_classes']
        backbone = config.get('backbone', 'resnet34')
        patch_size = config.get('patch_size', 64)
        num_patches = config.get('num_patches', 3)
        ablation_mode = config.get('ablation_mode', None)
        
        print(f"\nEvaluating {model_name} model...")
        print(f"  Config: backbone={backbone}, patches={num_patches}, patch_size={patch_size}, ablation={ablation_mode}")
        
        for input_size in input_sizes:
            print(f"  Input size: {input_size}x{input_size}")
            
            # Create model
            model = UGPL(
                num_classes=num_classes,
                input_size=input_size,
                patch_size=min(patch_size, input_size // 4),
                num_patches=num_patches,
                backbone=backbone,
                ablation_mode=ablation_mode
            ).to(device)
            
            # Count parameters
            params = count_parameters(model)
            print(f"  Parameters: {params:.2f}M")
            
            # Measure FLOPs
            input_shape = (1, 1, input_size, input_size)
            flops, _ = get_flops(model, input_shape, device)
            print(f"  GFLOPs: {flops:.2f}")
            
            # Measure inference time
            mean_time, std_time = measure_inference_time(
                model, device, input_shape, runs=50, return_std=True
            )
            print(f"  Inference time: {mean_time:.2f} ± {std_time:.2f} ms")
            
            # Measure memory usage
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                dummy_input = torch.randn(input_shape).to(device)
                _ = model(dummy_input)
            torch.cuda.synchronize()
            mem_used = torch.cuda.max_memory_allocated() / 1024**2
            
            # Add results
            results.append({
                'Model': model_name,
                'Memory Usage (MB)': mem_used,
                'Backbone': backbone,
                'Num Patches': num_patches,
                'Patch Size': patch_size,
                'Ablation': ablation_mode if ablation_mode else 'None',
                'Input Size': input_size,
                'Parameters (M)': params,
                'GFLOPs': flops,
                'Inference Time (ms)': mean_time,
                'Inference Std (ms)': std_time
            })
            
            # Free up memory
            del model
            torch.cuda.empty_cache()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def measure_component_metrics(components, input_sizes=(256,), device='cuda'):
    """Measure metrics for individual model components"""
    results = []
    
    for comp in components:
        model_name = comp['name']
        model = comp['model'].to(device)
        input_creator = comp['input_creator']
        
        print(f"\nEvaluating {model_name} component...")
        
        for input_size in input_sizes:
            print(f"  Input size: {input_size}x{input_size}")
            
            # Count parameters
            params = count_parameters(model)
            print(f"  Parameters: {params:.2f}M")
            
            # Create appropriate input shape
            input_shape = input_creator(input_size)
            
            # Create dummy input for inference
            dummy_input = tuple(torch.randn(shape).to(device) for shape in input_shape) \
                         if isinstance(input_shape, list) else torch.randn(input_shape).to(device)
            
            # Measure inference time
            model.eval()
            timings = []
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    if isinstance(dummy_input, tuple):
                        _ = model(*dummy_input)
                    else:
                        _ = model(dummy_input)
                    torch.cuda.synchronize()
                
                # Timed runs
                for _ in range(50):
                    start = time.perf_counter()
                    if isinstance(dummy_input, tuple):
                        _ = model(*dummy_input)
                    else:
                        _ = model(dummy_input)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    timings.append((end - start) * 1000)  # Convert to ms
            
            mean_time = np.mean(timings)
            std_time = np.std(timings)
            print(f"  Inference time: {mean_time:.2f} ± {std_time:.2f} ms")
            
            # Measure memory usage
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                if isinstance(dummy_input, tuple):
                    _ = model(*dummy_input)
                else:
                    _ = model(dummy_input)
            torch.cuda.synchronize()
            mem_used = torch.cuda.max_memory_allocated() / 1024**2
            
            # Add results
            results.append({
                'Component': model_name,
                'Input Size': input_size,
                'Parameters (M)': params,
                'Inference Time (ms)': mean_time,
                'Inference Std (ms)': std_time,
                'Memory Usage (MB)': mem_used,
            })
            
            # Free up memory
            torch.cuda.empty_cache()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def create_visualizations(full_results, component_results, save_dir='efficiency_metrics'):
    """Create visualizations of efficiency metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # 1. Parameters vs. Inference Time
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=full_results, 
        x='Parameters (M)', 
        y='Inference Time (ms)',
        hue='Model',
        style='Ablation',
        size='Input Size',
        sizes=(50, 200),
        alpha=0.7
    )
    plt.title('Model Parameters vs. Inference Time')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'params_vs_inference.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. GFLOPs vs. Inference Time
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=full_results, 
        x='GFLOPs', 
        y='Inference Time (ms)',
        hue='Model',
        style='Ablation',
        size='Input Size',
        sizes=(50, 200),
        alpha=0.7
    )
    plt.title('Computational Complexity (GFLOPs) vs. Inference Time')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gflops_vs_inference.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Inference Time by Backbone
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=full_results[full_results['Input Size'] == 256],  # Fixed input size for comparison
        x='Backbone',
        y='Inference Time (ms)',
        hue='Ablation',
        palette='viridis'
    )
    plt.title('Inference Time by Backbone and Ablation Mode (256x256 input)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'backbone_inference_times.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Inference Time by Number of Patches
    plt.figure(figsize=(12, 7))
    patch_data = full_results[(full_results['Ablation'] == 'None') & (full_results['Input Size'] == 256)]
    sns.barplot(
        data=patch_data,
        x='Num Patches',
        y='Inference Time (ms)',
        hue='Backbone',
        palette='rocket'
    )
    plt.title('Inference Time by Number of Patches and Backbone (256x256 input)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'patches_inference_times.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Component Inference Times
    if component_results is not None:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=component_results,
            x='Component',
            y='Inference Time (ms)',
            hue='Input Size',
            palette='viridis'
        )
        plt.title('Inference Time by Model Component')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'component_inference_times.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Component Parameters
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=component_results,
            x='Component',
            y='Parameters (M)',
            palette='rocket'
        )
        plt.title('Parameter Count by Model Component')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'component_parameters.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Memory Usage vs. Input Size
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=full_results,
        x='Input Size',
        y='Memory Usage (MB)',
        hue='Model',
        style='Ablation',
        marker='o',
        markersize=8
    )
    plt.title('Memory Usage vs. Input Size')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'memory_vs_input_size.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Ablation Impact on Efficiency (at 256x256)
    ablation_data = full_results[full_results['Input Size'] == 256]
    
    # Normalize data for comparison (relative to full model)
    datasets = ablation_data['Model'].unique()
    normalized_data = []
    
    for dataset in datasets:
        dataset_data = ablation_data[ablation_data['Model'] == dataset]
        full_model = dataset_data[dataset_data['Ablation'] == 'None']
        
        if len(full_model) == 0:
            continue
            
        full_time = full_model['Inference Time (ms)'].values[0]
        full_params = full_model['Parameters (M)'].values[0]
        full_flops = full_model['GFLOPs'].values[0]
        
        for _, row in dataset_data.iterrows():
            normalized_data.append({
                'Model': row['Model'],
                'Ablation': row['Ablation'],
                'Relative Time': row['Inference Time (ms)'] / full_time,
                'Relative Params': row['Parameters (M)'] / full_params,
                'Relative GFLOPs': row['GFLOPs'] / full_flops
            })
    
    normalized_df = pd.DataFrame(normalized_data)
    
    # Melt the dataframe for easier plotting
    melted_df = normalized_df.melt(
        id_vars=['Model', 'Ablation'],
        value_vars=['Relative Time', 'Relative Params', 'Relative GFLOPs'],
        var_name='Metric',
        value_name='Relative Value'
    )
    
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=melted_df,
        x='Ablation',
        y='Relative Value',
        hue='Metric',
        palette='Set2'
    )
    plt.axhline(y=1.0, color='red', linestyle='--')
    plt.title('Relative Efficiency Impact of Ablations (256x256 input)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_efficiency_impact.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    