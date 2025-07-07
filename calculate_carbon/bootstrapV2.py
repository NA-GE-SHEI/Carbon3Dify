import os
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from scipy import stats
import pandas as pd
from pathlib import Path
import warnings
import matplotlib


# Fix matplotlib boxplot labels parameter warning
warnings.filterwarnings("ignore", 
                      message="The 'labels' parameter of boxplot.*", 
                      category=matplotlib.MatplotlibDeprecationWarning)


print(f"Matplotlib {matplotlib.__version__} - boxplot labels warning suppressed")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Bootstrap Analysis Tool - Analyze Chair Weight Prediction Results')
    
    # Main parameters
    parser.add_argument('--n_iterations', type=int, default=100, 
                        help='Number of times to execute xg&rf&mlr&svr_analysis.py')
    parser.add_argument('--result_dir', type=str, default='./result', 
                        help='Results directory path')
    parser.add_argument('--output_dir', type=str, default='./bootstrap_analysis', 
                        help='Analysis results output directory')
    parser.add_argument('--ci_level', type=float, default=0.95, 
                        help='Confidence interval level (default: 95%)')
    parser.add_argument('--min_num', type=int, default=10, 
                        help='Only analyze result_{num} folders where num is greater than this value')
    parser.add_argument('--py_path', type=str, default='xg&rf&mlr&svr_analysis.py', 
                        help='Path to the .py file')
    
    # xg&rf&mlr&svr_analysis.py parameters
    parser.add_argument('--data_file', type=str, default='chair_raw_data_v1.5.csv', 
                        help='Data file path (passed to analysis script)')
    parser.add_argument('--encoding', type=str, default='big5', 
                        help='CSV file encoding (passed to analysis script)')
    parser.add_argument('--model_type', type=str, default='all', 
                        choices=['rf', 'xgb', 'mlr', 'svr', 'all'], 
                        help='Model type (passed to analysis script)')
    
    # SVR specific parameters
    parser.add_argument('--svr_kernel', type=str, default='rbf', 
                        choices=['linear', 'poly', 'rbf', 'sigmoid'], 
                        help='SVR kernel type (passed to analysis script)')
    parser.add_argument('--svr_c', type=float, default=1.0, 
                        help='SVR regularization parameter (passed to analysis script)')
    parser.add_argument('--svr_epsilon', type=float, default=0.1, 
                        help='SVR epsilon parameter (passed to analysis script)')
    
    # Visualization parameters
    parser.add_argument('--dpi', type=int, default=300, 
                        help='Image resolution')
    
    return parser.parse_args()


def setup_output_dir(dir_path):
    """Create output directory"""
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def run_analysis_script(args, iteration):
    """Execute analysis script and collect results"""
    cmd = [
        'python3.10', args.py_path,
        '--data_file', args.data_file,
        '--encoding', args.encoding,
        '--model_type', args.model_type,
        # Use time + iteration as random seed to ensure different results each time
        '--random_state', str(int(time.time()) % 10000 + iteration)
    ]
    
    # Add SVR specific parameters if using SVR model
    if args.model_type in ['svr', 'all']:
        cmd.extend(['--svr_kernel', args.svr_kernel])
        cmd.extend(['--svr_c', str(args.svr_c)])
        cmd.extend(['--svr_epsilon', str(args.svr_epsilon)])
    
    try:
        print(f"Executing iteration {iteration+1}/{args.n_iterations}...")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing analysis script: {e}")
        return False


def extract_r2_values(result_dir, min_num=10):
    """Extract R² values from results.txt files"""
    rf_r2_values = []
    xgb_r2_values = []
    mlr_r2_values = []  # Added: Multiple Linear Regression
    svr_r2_values = []  # Added: Support Vector Regression
    folder_nums = []
    
    result_path = Path(result_dir)
    result_folders = [f for f in result_path.glob('result_*') if f.is_dir()]
    
    # Regular expression patterns
    rf_pattern = r"隨機森林 R²:\s*([\d\.]+)"
    xgb_pattern = r"XGBoost R²:\s*([\d\.]+)"
    mlr_pattern = r"多元線性迴歸 R²:\s*([\d\.]+)"  # Added pattern
    svr_pattern = r"支持向量迴歸 R²:\s*([\d\.]+)"  # Added pattern
    
    for folder in result_folders:
        try:
            # Extract folder number
            folder_num = int(folder.name.split('_')[1])
            
            # Only process folders with numbers greater than min_num
            if folder_num <= min_num:
                continue
                
            result_file = folder / 'results.txt'
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    has_data = False
                    
                    # Extract Random Forest R²
                    rf_match = re.search(rf_pattern, content)
                    if rf_match:
                        rf_r2 = float(rf_match.group(1))
                        rf_r2_values.append(rf_r2)
                        has_data = True
                    
                    # Extract XGBoost R²
                    xgb_match = re.search(xgb_pattern, content)
                    if xgb_match:
                        xgb_r2 = float(xgb_match.group(1))
                        xgb_r2_values.append(xgb_r2)
                        has_data = True
                    
                    # Extract Multiple Linear Regression R²
                    mlr_match = re.search(mlr_pattern, content)
                    if mlr_match:
                        mlr_r2 = float(mlr_match.group(1))
                        mlr_r2_values.append(mlr_r2)
                        has_data = True
                    
                    # Extract Support Vector Regression R²
                    svr_match = re.search(svr_pattern, content)
                    if svr_match:
                        svr_r2 = float(svr_match.group(1))
                        svr_r2_values.append(svr_r2)
                        has_data = True
                    
                    # Record folder number if any value was successfully extracted
                    if has_data:
                        folder_nums.append(folder_num)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
    
    return {
        'rf_r2': rf_r2_values,
        'xgb_r2': xgb_r2_values,
        'mlr_r2': mlr_r2_values,  # Added
        'svr_r2': svr_r2_values,  # Added
        'folder_nums': folder_nums
    }


def calculate_statistics(r2_values, ci_level=0.95):
    """Calculate R² statistics"""
    if not r2_values or len(r2_values) == 0:
        return None
    
    # Convert to numpy array
    r2_array = np.array(r2_values)
    
    # Calculate basic statistics
    mean_r2 = np.mean(r2_array)
    median_r2 = np.median(r2_array)
    std_r2 = np.std(r2_array)
    min_r2 = np.min(r2_array)
    max_r2 = np.max(r2_array)
    
    # Calculate confidence interval
    ci_lower = np.percentile(r2_array, (1 - ci_level) * 100 / 2)
    ci_upper = np.percentile(r2_array, 100 - (1 - ci_level) * 100 / 2)
    
    # Calculate standard error
    se = std_r2 / np.sqrt(len(r2_array))
    
    # Calculate parametric confidence interval using t-distribution
    t_ci_lower, t_ci_upper = stats.t.interval(
        ci_level, 
        len(r2_array) - 1, 
        loc=mean_r2, 
        scale=se
    )
    
    return {
        'mean': mean_r2,
        'median': median_r2,
        'std': std_r2,
        'min': min_r2,
        'max': max_r2,
        'bootstrap_ci': (ci_lower, ci_upper),
        'parametric_ci': (t_ci_lower, t_ci_upper),
        'n_samples': len(r2_array),
        'se': se
    }


def visualize_r2_distribution(r2_values, stats, model_name, output_dir, args):
    """Visualize R² distribution"""
    if not r2_values or len(r2_values) == 0:
        print(f"Insufficient {model_name} R² data to plot distribution")
        return
    
    plt.figure(figsize=(12, 7))
    
    # Set font for English display
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Plot histogram and KDE
    sns.histplot(r2_values, kde=True, color='steelblue')
    
    # Add mean line
    plt.axvline(x=stats['mean'], color='red', linestyle='--', 
               label=f'Mean: {stats["mean"]:.4f}')
    
    # Add median line
    plt.axvline(x=stats['median'], color='green', linestyle='-', 
               label=f'Median: {stats["median"]:.4f}')
    
    # Add confidence interval
    plt.axvline(x=stats['bootstrap_ci'][0], color='orange', linestyle=':', 
               label=f'Bootstrap {args.ci_level*100:.0f}% CI: [{stats["bootstrap_ci"][0]:.4f}, {stats["bootstrap_ci"][1]:.4f}]')
    plt.axvline(x=stats['bootstrap_ci'][1], color='orange', linestyle=':')
    
    # Chart title and axis labels
    plt.title(f'{model_name} R² Distribution (n={stats["n_samples"]})', fontsize=22)
    plt.xlabel('R² Value', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    
    # Add statistics text box
    stats_text = (
        f"Basic Statistics:\n"
        f"Mean: {stats['mean']:.6f}\n"
        f"Median: {stats['median']:.6f}\n"
        f"Std Dev: {stats['std']:.6f}\n"
        f"Min: {stats['min']:.6f}\n"
        f"Max: {stats['max']:.6f}\n"
        f"Std Error: {stats['se']:.6f}\n"
        f"Bootstrap {args.ci_level*100:.0f}% CI: [{stats['bootstrap_ci'][0]:.6f}, {stats['bootstrap_ci'][1]:.6f}]\n"
        f"Parametric {args.ci_level*100:.0f}% CI: [{stats['parametric_ci'][0]:.6f}, {stats['parametric_ci'][1]:.6f}]"
    )
    plt.xlim(0.5, 1.0)   # Fix X-axis
    plt.ylim(0, 80)  # Set Y-axis if consistency needed, otherwise comment out    
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                va='top', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()

    # Save image
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_r2_distribution.png'), dpi=args.dpi)
    plt.close()


def visualize_r2_comparison(model_stats, output_dir, args):
    """Visualize R² comparison across all models"""
    # Filter valid model statistics
    valid_models = {name: stats for name, stats in model_stats.items() if stats is not None}
    
    if len(valid_models) < 2:
        print("Insufficient model data to plot comparison chart")
        return
    
    # ====== Added: Detailed statistical information output ======
    print("\n" + "="*80)
    print("Model R² Comparison (with Confidence Intervals) - Detailed Statistical Results")
    print("="*80)
    
    for model_name, stats in valid_models.items():
        print(f"\n【{model_name}】")
        print(f"  Sample Size: {stats['n_samples']}")
        print(f"  Mean R²: {stats['mean']:.6f}")
        print(f"  Median R²: {stats['median']:.6f}")
        print(f"  Standard Deviation: {stats['std']:.6f}")
        print(f"  Standard Error: {stats['se']:.6f}")
        print(f"  Minimum: {stats['min']:.6f}")
        print(f"  Maximum: {stats['max']:.6f}")
        print(f"  Bootstrap {args.ci_level*100:.0f}% Confidence Interval: [{stats['bootstrap_ci'][0]:.6f}, {stats['bootstrap_ci'][1]:.6f}]")
        print(f"  CI Width: {stats['bootstrap_ci'][1] - stats['bootstrap_ci'][0]:.6f}")
        print(f"  Parametric {args.ci_level*100:.0f}% Confidence Interval: [{stats['parametric_ci'][0]:.6f}, {stats['parametric_ci'][1]:.6f}]")
        print(f"  Parametric CI Width: {stats['parametric_ci'][1] - stats['parametric_ci'][0]:.6f}")
    
    # Inter-model comparison analysis
    print(f"\n【Inter-Model Comparison Analysis】")
    models = list(valid_models.keys())
    print(f"Models in comparison: {', '.join(models)}")
    
    # Calculate and display differences between all model pairs
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1 = models[i]
            model2 = models[j]
            mean_diff = valid_models[model1]['mean'] - valid_models[model2]['mean']
            
            # Check if confidence intervals overlap
            ci1_lower, ci1_upper = valid_models[model1]['bootstrap_ci']
            ci2_lower, ci2_upper = valid_models[model2]['bootstrap_ci']
            overlap = max(0, min(ci1_upper, ci2_upper) - max(ci1_lower, ci2_lower))
            ci_overlap = overlap > 0
            
            print(f"\n  {model1} vs {model2}:")
            print(f"    Mean R² Difference: {mean_diff:.6f}")
            print(f"    Relative Difference: {(mean_diff/valid_models[model2]['mean'])*100:.2f}%")
            print(f"    CI Overlap: {'Yes' if ci_overlap else 'No'}")
            if ci_overlap:
                print(f"    Overlap Length: {overlap:.6f}")
    
    # Ranking analysis
    print(f"\n【Model Ranking (by Mean R² Value)】")
    sorted_models = sorted(valid_models.items(), key=lambda x: x[1]['mean'], reverse=True)
    for rank, (model_name, stats) in enumerate(sorted_models, 1):
        print(f"  Rank {rank}: {model_name} (R² = {stats['mean']:.6f})")
    
    print("="*80)
    # ====== Original plotting code ======
    
    plt.figure(figsize=(12, 8))
    
    # Set font for English display
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Prepare data
    models = list(valid_models.keys())
    means = [valid_models[model]['mean'] for model in models]
    
    # Calculate error bars
    errors = []
    for model in models:
        stats = valid_models[model]
        lower_error = means[models.index(model)] - stats['bootstrap_ci'][0]
        upper_error = stats['bootstrap_ci'][1] - means[models.index(model)]
        errors.append([lower_error, upper_error])
    
    # Set color mapping
    model_colors = {
        'Random Forest': 'steelblue',
        'XGBoost': 'darkorange',
        'Multiple Linear Regression': 'forestgreen',
        'Support Vector Regression': 'firebrick'
    }
    
    colors = [model_colors.get(model, 'gray') for model in models]
    
    # Plot bar chart
    bars = plt.bar(models, means, color=colors)
    
    # Add error bars
    plt.errorbar(models, means, yerr=np.array(errors).T, fmt='o', color='black', 
                ecolor='black', elinewidth=2, capsize=6)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{means[i]:.4f}', ha='center', va='bottom', fontsize=11)
    
    # Chart title and axis labels
    plt.title('Model R² Comparison (with Confidence Intervals)', fontsize=16)
    plt.ylabel('R² Value', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis range to ensure error bars don't exceed chart boundaries
    all_ci_values = []
    for model in models:
        all_ci_values.extend(valid_models[model]['bootstrap_ci'])
    
    min_val = min(all_ci_values) - 0.05
    max_val = max(all_ci_values) + 0.05
    plt.ylim(min_val, max_val)
    
    plt.tight_layout()
    
    # Save image
    plt.savefig(os.path.join(output_dir, 'model_r2_comparison.png'), dpi=args.dpi)
    plt.close()


def create_compatible_boxplot(data, labels=None, **kwargs):
    """
    Create matplotlib version-compatible boxplot function
    Fix labels parameter warning issue
    """
    try:
        from packaging import version
        matplotlib_version = version.parse(matplotlib.__version__)
        
        # Matplotlib 3.9+ uses tick_labels parameter
        if matplotlib_version >= version.parse("3.9.0"):
            if labels is not None:
                kwargs['tick_labels'] = labels
            return plt.boxplot(data, **kwargs)
        else:
            # Older versions use labels parameter
            if labels is not None:
                kwargs['labels'] = labels
            return plt.boxplot(data, **kwargs)
    except ImportError:
        # If packaging library is not available, use labels directly (warning but functional)
        if labels is not None:
            kwargs['labels'] = labels
        return plt.boxplot(data, **kwargs)


def visualize_r2_boxplot(r2_data, model_names, output_dir, args):
    """Visualize R² boxplot comparison (fixed version)"""
    # Define model name to r2_data key mapping
    model_key_mapping = {
        'Random Forest': 'rf',
        'XGBoost': 'xgb',
        'Multiple Linear Regression': 'mlr',
        'Support Vector Regression': 'svr'
    }
    
    # Filter valid model data
    valid_data = []
    valid_labels = []
    valid_data_dict = {}
    
    for model in model_names:
        model_key = model_key_mapping.get(model)
        if model_key and f'{model_key}_r2' in r2_data and r2_data[f'{model_key}_r2'] and len(r2_data[f'{model_key}_r2']) > 0:
            data = r2_data[f'{model_key}_r2']
            valid_data.append(data)
            valid_labels.append(model)
            valid_data_dict[model] = data
    
    if len(valid_data) < 2:
        print("Insufficient data to plot boxplot")
        return
    
    # ====== Added: Detailed boxplot statistical information output ======
    print("\n" + "="*80)
    print("Model R² Distribution Comparison (Boxplot Analysis) - Detailed Statistical Results")
    print("="*80)
    
    for model in valid_labels:
        data = valid_data_dict[model]
        data_array = np.array(data)
        
        # Calculate boxplot statistics
        q1 = np.percentile(data_array, 25)
        q2 = np.percentile(data_array, 50)  # median
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        
        # Calculate outlier boundaries
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        # Find outliers
        outliers = data_array[(data_array < lower_fence) | (data_array > upper_fence)]
        
        # Calculate boxplot whiskers
        whisker_low = np.min(data_array[data_array >= lower_fence]) if len(data_array[data_array >= lower_fence]) > 0 else np.min(data_array)
        whisker_high = np.max(data_array[data_array <= upper_fence]) if len(data_array[data_array <= upper_fence]) > 0 else np.max(data_array)
        
        print(f"\n【{model}】(Sample Size: {len(data)})")
        print(f"  Five-Number Summary:")
        print(f"    Minimum: {np.min(data_array):.6f}")
        print(f"    First Quartile (Q1): {q1:.6f}")
        print(f"    Median (Q2): {q2:.6f}")
        print(f"    Third Quartile (Q3): {q3:.6f}")
        print(f"    Maximum: {np.max(data_array):.6f}")
        print(f"  Distribution Characteristics:")
        print(f"    Interquartile Range (IQR): {iqr:.6f}")
        print(f"    Lower Whisker: {whisker_low:.6f}")
        print(f"    Upper Whisker: {whisker_high:.6f}")
        print(f"    Data Range: {np.max(data_array) - np.min(data_array):.6f}")
        print(f"  Outlier Analysis:")
        print(f"    Number of Outliers: {len(outliers)}")
        print(f"    Outlier Percentage: {len(outliers)/len(data)*100:.2f}%")
        if len(outliers) > 0:
            print(f"    Outliers: {', '.join([f'{x:.6f}' for x in sorted(outliers)])}")
        
        # Calculate skewness and kurtosis
        from scipy import stats
        skewness = stats.skew(data_array)
        kurtosis = stats.kurtosis(data_array)
        print(f"  Distribution Shape:")
        print(f"    Skewness: {skewness:.4f} ({'Right-skewed' if skewness > 0 else 'Left-skewed' if skewness < 0 else 'Symmetric'})")
        print(f"    Kurtosis: {kurtosis:.4f} ({'High peak' if kurtosis > 0 else 'Low peak' if kurtosis < 0 else 'Normal peak'})")
    
    # Inter-model distribution comparison
    print(f"\n【Inter-Model Distribution Comparison】")
    for i in range(len(valid_labels)):
        for j in range(i+1, len(valid_labels)):
            model1, model2 = valid_labels[i], valid_labels[j]
            data1, data2 = valid_data_dict[model1], valid_data_dict[model2]
            
            # Median difference
            median_diff = np.median(data1) - np.median(data2)
            
            # IQR comparison
            iqr1 = np.percentile(data1, 75) - np.percentile(data1, 25)
            iqr2 = np.percentile(data2, 75) - np.percentile(data2, 25)
            iqr_ratio = iqr1 / iqr2 if iqr2 != 0 else float('inf')
            
            print(f"\n  {model1} vs {model2}:")
            print(f"    Median Difference: {median_diff:.6f}")
            print(f"    IQR Ratio: {iqr_ratio:.3f} ({model1} variability {'higher' if iqr_ratio > 1 else 'lower'})")
            
            # Perform Mann-Whitney U test (non-parametric test)
            try:
                u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                print(f"    Mann-Whitney U Test: U={u_stat:.2f}, p={p_value:.6f}")
                print(f"    Conclusion: {'Significant' if p_value < 0.05 else 'No significant'} distribution difference (α=0.05)")
            except Exception as e:
                print(f"    Mann-Whitney U Test failed: {e}")
    
    # Distribution consistency analysis
    print(f"\n【Distribution Stability Ranking】")
    stability_scores = []
    for model in valid_labels:
        data = valid_data_dict[model]
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        cv = np.std(data) / np.mean(data)  # coefficient of variation
        stability_score = 1 / (iqr + cv)  # stability score (higher = more stable)
        stability_scores.append((model, stability_score, iqr, cv))
    
    stability_scores.sort(key=lambda x: x[1], reverse=True)
    for rank, (model, score, iqr, cv) in enumerate(stability_scores, 1):
        print(f"  Rank {rank}: {model} (IQR={iqr:.6f}, CV={cv:.4f})")
    
    print("="*80)
    # ====== Fixed plotting code ======
    
    plt.figure(figsize=(10, 7))
    
    # Set font for English display
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Set color mapping
    model_colors = {
        'Random Forest': 'steelblue',
        'XGBoost': 'darkorange',
        'Multiple Linear Regression': 'forestgreen',
        'Support Vector Regression': 'firebrick'
    }
    
    # Use fixed compatibility boxplot function
    box = create_compatible_boxplot(valid_data, labels=valid_labels, patch_artist=True)
    
    # Set box colors
    colors = [model_colors.get(model, 'gray') for model in valid_labels]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Chart title and axis labels
    plt.title('Model R² Distribution Comparison', fontsize=16)
    plt.ylabel('R² Value', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save image
    plt.savefig(os.path.join(output_dir, 'model_r2_boxplot.png'), dpi=args.dpi)
    plt.close()


def visualize_r2_trend(folder_nums, r2_data, model_names, output_dir, args):
    """Visualize R² trend over iterations"""
    if not folder_nums or len(folder_nums) == 0:
        print("Insufficient data to plot trend chart")
        return
    
    # Define model name to r2_data key mapping
    model_key_mapping = {
        'Random Forest': 'rf',
        'XGBoost': 'xgb',
        'Multiple Linear Regression': 'mlr',
        'Support Vector Regression': 'svr'
    }
    
    # Set color and marker mapping
    model_colors = {
        'Random Forest': ('steelblue', 'o-', 'navy'),
        'XGBoost': ('darkorange', 's-', 'darkred'),
        'Multiple Linear Regression': ('forestgreen', '^-', 'darkgreen'),
        'Support Vector Regression': ('firebrick', 'd-', 'maroon')
    }
    
    # Create dictionary to map folder numbers to R² values
    model_data = {}
    for model in model_names:
        model_key = model_key_mapping.get(model)
        if model_key and f'{model_key}_r2' in r2_data and r2_data[f'{model_key}_r2']:
            model_data[model] = {}
            for i, folder in enumerate(folder_nums):
                if i < len(r2_data[f'{model_key}_r2']):
                    model_data[model][folder] = r2_data[f'{model_key}_r2'][i]
    
    # Check if there's valid data
    if not model_data:
        print("No valid model data to plot trend chart")
        return
    
    # Sort folder numbers
    sorted_folders = sorted(set(folder_nums))
    
    plt.figure(figsize=(14, 8))
    
    # Set font for English display
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Plot trend lines for each model
    for model in model_data:
        if not model_data[model]:
            continue
            
        # Prepare plotting data
        model_folders = []
        model_values = []
        for folder in sorted_folders:
            if folder in model_data[model]:
                model_folders.append(folder)
                model_values.append(model_data[model][folder])
        
        if not model_folders:
            continue
            
        color, marker, ma_color = model_colors.get(model, ('gray', '*-', 'dimgray'))
        
        # Plot R² trend
        plt.plot(model_folders, model_values, marker, color=color, label=f'{model} R²')
        
        # Plot moving average line (if enough points)
        window_size = min(5, len(model_folders) // 2) if len(model_folders) > 10 else 2
        if window_size > 1:
            model_ma = pd.Series(model_values).rolling(window=window_size).mean()
            plt.plot(model_folders, model_ma, '--', color=ma_color, 
                   label=f'{model} {window_size}-point Moving Average')
    
    # Chart title and axis labels
    plt.title('R² Trend Over Iterations', fontsize=16)
    plt.xlabel('Iteration Number', fontsize=14)
    plt.ylabel('R² Value', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save image
    plt.savefig(os.path.join(output_dir, 'r2_trend_by_folder.png'), dpi=args.dpi)
    plt.close()


def save_statistics(model_stats, r2_data, output_dir, args):
    """Save statistical results"""
    # Define model name to r2_data key mapping
    model_key_mapping = {
        'Random Forest': 'rf',
        'XGBoost': 'xgb',
        'Multiple Linear Regression': 'mlr',
        'Support Vector Regression': 'svr'
    }
    
    with open(os.path.join(output_dir, 'bootstrap_statistics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Bootstrap Statistical Analysis Results (Confidence Level: {args.ci_level*100}%)\n\n")
        
        # Save statistical results for each model
        for model_name, stats in model_stats.items():
            if stats:
                f.write(f"{model_name} Model R² Statistics (Sample Size: {stats['n_samples']}):\n")
                f.write(f"  Mean: {stats['mean']:.6f}\n")
                f.write(f"  Median: {stats['median']:.6f}\n")
                f.write(f"  Standard Deviation: {stats['std']:.6f}\n")
                f.write(f"  Standard Error: {stats['se']:.6f}\n")
                f.write(f"  Minimum: {stats['min']:.6f}\n")
                f.write(f"  Maximum: {stats['max']:.6f}\n")
                f.write(f"  Bootstrap {args.ci_level*100}% Confidence Interval: [{stats['bootstrap_ci'][0]:.6f}, {stats['bootstrap_ci'][1]:.6f}]\n")
                f.write(f"  Parametric {args.ci_level*100}% Confidence Interval: [{stats['parametric_ci'][0]:.6f}, {stats['parametric_ci'][1]:.6f}]\n")
                f.write("\n")
        
        # Save model comparisons
        valid_models = {name: stats for name, stats in model_stats.items() if stats is not None}
        if len(valid_models) >= 2:
            f.write("Model Comparisons:\n")
            
            # Calculate differences between all model pairs
            model_names = list(valid_models.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]
                    mean_diff = valid_models[model1]['mean'] - valid_models[model2]['mean']
                    f.write(f"  R² Mean Difference ({model1} - {model2}): {mean_diff:.6f}\n")
            
            f.write("\n")
            
            # Attempt t-test
            try:
                # Create dictionary for each model for paired data
                model_dicts = {}
                for model in valid_models:
                    model_key = model_key_mapping.get(model, model.lower().replace(" ", "_"))
                    r2_key = f'{model_key}_r2'
                    
                    if r2_key in r2_data and r2_data[r2_key]:
                        model_dicts[model] = {
                            folder: r2 
                            for folder, r2 in zip(
                                r2_data['folder_nums'][:len(r2_data[r2_key])], 
                                r2_data[r2_key]
                            )
                        }
                
                # Perform t-test for each model pair
                for i in range(len(model_names)):
                    for j in range(i+1, len(model_names)):
                        model1 = model_names[i]
                        model2 = model_names[j]
                        
                        if model1 in model_dicts and model2 in model_dicts:
                            # Find common folders
                            common_folders = set(model_dicts[model1].keys()) & set(model_dicts[model2].keys())
                            
                            if common_folders:
                                values1 = [model_dicts[model1][folder] for folder in common_folders]
                                values2 = [model_dicts[model2][folder] for folder in common_folders]
                                
                                # Use paired data for t-test
                                t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                                f.write(f"  {model1} vs {model2} Independent Samples t-test Results (based on {len(common_folders)} paired samples):\n")
                                f.write(f"    t = {t_stat:.4f}, p = {p_value:.6f}\n")
                                f.write(f"    Conclusion: {'Statistically significant' if p_value < 0.05 else 'No statistically significant'} difference (α = 0.05)\n\n")
                            else:
                                f.write(f"  {model1} vs {model2}: Cannot perform t-test - insufficient paired data\n\n")
            except Exception as e:
                f.write(f"  t-test calculation error: {e}\n\n")
    
    # Save raw R² data for further analysis
    try:
        # Create DataFrame to store all model R² values
        result_data = []
        folder_nums_set = set(r2_data['folder_nums'])
        
        # Prepare model data dictionaries
        model_dicts = {}
        model_names = ['Random Forest', 'XGBoost', 'Multiple Linear Regression', 'Support Vector Regression']
        
        for model in model_names:
            model_key = model_key_mapping.get(model, model.lower().replace(" ", "_"))
            r2_key = f'{model_key}_r2'
            
            if r2_key in r2_data and r2_data[r2_key]:
                model_dicts[model] = {}
                for i, folder in enumerate(r2_data['folder_nums']):
                    if i < len(r2_data[r2_key]):
                        model_dicts[model][folder] = r2_data[r2_key][i]
        
        # Create one row of data for each folder
        for folder in sorted(folder_nums_set):
            row = {'folder_num': folder}
            for model in model_names:
                if model in model_dicts and folder in model_dicts[model]:
                    row[f'{model.lower().replace(" ", "_")}_r2'] = model_dicts[model][folder]
            result_data.append(row)
        
        # Create DataFrame and save
        pd.DataFrame(result_data).to_csv(os.path.join(output_dir, 'r2_values.csv'), index=False)
    except Exception as e:
        print(f"Error saving R² values CSV: {e}")
        print("Attempting to save individual model R² values...")
        
        # Save data for each model separately
        for model in model_names:
            model_key = model_key_mapping.get(model, model.lower().replace(" ", "_"))
            r2_key = f'{model_key}_r2'
            
            if r2_key in r2_data and r2_data[r2_key]:
                pd.DataFrame({
                    f'{model_key}_r2': r2_data[r2_key]
                }).to_csv(os.path.join(output_dir, f'{model_key}_r2_values.csv'), index=False)


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    output_dir = setup_output_dir(args.output_dir)
    print(f"Saving analysis results to: {output_dir}")
    
    # Check if analysis script needs to be executed
    need_to_run_analysis = args.n_iterations > 0
    
    if need_to_run_analysis:
        print(f"Will execute {args.py_path} {args.n_iterations} times...")
        for i in range(args.n_iterations):
            success = run_analysis_script(args, i)
            if not success:
                print(f"Warning: Iteration {i+1} failed")
    else:
        print(f"Skipping execution of {args.py_path}, analyzing existing results...")
    
    # Extract result data
    print(f"Extracting R² values from {args.result_dir}, analyzing only folders with number > {args.min_num}...")
    r2_data = extract_r2_values(args.result_dir, args.min_num)
    
    # Check if there's sufficient data for analysis
    model_keys = ['rf_r2', 'xgb_r2', 'mlr_r2', 'svr_r2']
    if not any(r2_data[key] for key in model_keys):
        print(f"Error: No R² values found. Please check {args.result_dir} directory and folder number requirements.")
        return
    
    # Display found data quantities
    print(f"Found R² values for the following models:")
    print(f"  Random Forest: {len(r2_data['rf_r2'])} values")
    print(f"  XGBoost: {len(r2_data['xgb_r2'])} values")
    print(f"  Multiple Linear Regression: {len(r2_data['mlr_r2'])} values")
    print(f"  Support Vector Regression: {len(r2_data['svr_r2'])} values")
    
    # Calculate statistics
    print(f"Calculating statistics using {args.ci_level*100}% confidence level...")
    model_stats = {
        'Random Forest': calculate_statistics(r2_data['rf_r2'], args.ci_level) if r2_data['rf_r2'] else None,
        'XGBoost': calculate_statistics(r2_data['xgb_r2'], args.ci_level) if r2_data['xgb_r2'] else None,
        'Multiple Linear Regression': calculate_statistics(r2_data['mlr_r2'], args.ci_level) if r2_data['mlr_r2'] else None,
        'Support Vector Regression': calculate_statistics(r2_data['svr_r2'], args.ci_level) if r2_data['svr_r2'] else None
    }
    
    # Visualize R² distributions
    print("Generating visualization charts...")
    # Define model name to r2_data key mapping
    model_key_mapping = {
        'Random Forest': 'rf',
        'XGBoost': 'xgb',
        'Multiple Linear Regression': 'mlr',
        'Support Vector Regression': 'svr'
    }

    for model_name, stats in model_stats.items():
        if stats:
            model_key = model_key_mapping.get(model_name)
            if model_key and f'{model_key}_r2' in r2_data:
                visualize_r2_distribution(r2_data[f'{model_key}_r2'], stats, model_name, output_dir, args)
            else:
                print(f"Warning: Cannot find R² data for {model_name}")
    
    # Visualize model comparisons
    valid_models = {name: stats for name, stats in model_stats.items() if stats is not None}
    if len(valid_models) >= 2:
        visualize_r2_comparison(valid_models, output_dir, args)
        visualize_r2_boxplot(r2_data, ['Random Forest', 'XGBoost', 'Multiple Linear Regression', 'Support Vector Regression'], output_dir, args)
    
    # Visualize trends
    if len(r2_data['folder_nums']) > 1:
        visualize_r2_trend(r2_data['folder_nums'], r2_data, ['Random Forest', 'XGBoost', 'Multiple Linear Regression', 'Support Vector Regression'], output_dir, args)
    
    # Save statistical results
    print("Saving statistical results...")
    save_statistics(model_stats, r2_data, output_dir, args)
    
    print(f"\nBootstrap analysis complete! All results saved to: {output_dir}")
    
    # Print summary statistics
    for model_name, stats in model_stats.items():
        if stats:
            print(f"\n{model_name} R² Statistics (n={stats['n_samples']}):")
            print(f"  Mean: {stats['mean']:.4f}, {args.ci_level*100}% CI: [{stats['bootstrap_ci'][0]:.4f}, {stats['bootstrap_ci'][1]:.4f}]")
    
    # Print model comparisons
    valid_models = {name: stats for name, stats in model_stats.items() if stats is not None}
    if len(valid_models) >= 2:
        model_names = list(valid_models.keys())
        print(f"\nModel Comparisons:")
        
        # Display differences between all model pairs
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                mean_diff = valid_models[model1]['mean'] - valid_models[model2]['mean']
                print(f"  R² Mean Difference ({model1} - {model2}): {mean_diff:.4f}")
        
        # Attempt t-test analysis
        try:
            # Create dictionary for each model
            model_dicts = {}
            for model in valid_models:
                model_key = model.lower().replace(" ", "_")
                r2_key = f'{model_key}_r2'
                
                if r2_key in r2_data and r2_data[r2_key]:
                    model_dicts[model] = {
                        folder: r2 
                        for folder, r2 in zip(
                            r2_data['folder_nums'][:len(r2_data[r2_key])], 
                            r2_data[r2_key]
                        )
                    }
            
            # Perform t-test for each model pair
            print("\nInter-model t-test Results:")
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]
                    
                    if model1 in model_dicts and model2 in model_dicts:
                        # Find common folders
                        common_folders = set(model_dicts[model1].keys()) & set(model_dicts[model2].keys())
                        
                        if common_folders:
                            values1 = [model_dicts[model1][folder] for folder in common_folders]
                            values2 = [model_dicts[model2][folder] for folder in common_folders]
                            
                            # Use paired data for t-test
                            t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                            print(f"  {model1} vs {model2} (based on {len(common_folders)} samples):")
                            print(f"    t={t_stat:.4f}, p={p_value:.4f}")
                            print(f"    Conclusion: {'Statistically significant' if p_value < 0.05 else 'No statistically significant'} difference (α=0.05)")
                        else:
                            print(f"  {model1} vs {model2}: Cannot perform t-test - insufficient paired data")
        except Exception as e:
            print(f"  t-test calculation error: {e}")


if __name__ == "__main__":
    main()