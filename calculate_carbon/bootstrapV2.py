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

# 修復 matplotlib boxplot labels 參數警告
warnings.filterwarnings("ignore", 
                      message="The 'labels' parameter of boxplot.*", 
                      category=matplotlib.MatplotlibDeprecationWarning)

print(f"Matplotlib {matplotlib.__version__} - 已抑制boxplot labels警告")

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='Bootstrap分析工具 - 分析椅子重量預測結果')
    
    # 主要參數
    parser.add_argument('--n_iterations', type=int, default=100, 
                        help='執行xg&rf&mlr&svr_analysis.py的次數')
    parser.add_argument('--result_dir', type=str, default='./result', 
                        help='結果目錄路徑')
    parser.add_argument('--output_dir', type=str, default='./bootstrap_analysis', 
                        help='分析結果輸出目錄')
    parser.add_argument('--ci_level', type=float, default=0.95, 
                        help='信賴區間水平 (默認: 95%)')
    parser.add_argument('--min_num', type=int, default=10, 
                        help='僅分析result_{num}文件夾，其中num大於此值')
    parser.add_argument('--py_path', type=str, default='xg&rf&mlr&svr_analysis.py', 
                        help='.py檔案的路徑')
    
    # xg&rf&mlr&svr_analysis.py的參數
    parser.add_argument('--data_file', type=str, default='chair_raw_data_v1.5.csv', 
                        help='資料檔案路徑 (傳給分析腳本)')
    parser.add_argument('--encoding', type=str, default='big5', 
                        help='CSV檔案編碼 (傳給分析腳本)')
    parser.add_argument('--model_type', type=str, default='all', 
                        choices=['rf', 'xgb', 'mlr', 'svr', 'all'], 
                        help='模型類型 (傳給分析腳本)')
    
    # SVR特定參數
    parser.add_argument('--svr_kernel', type=str, default='rbf', 
                        choices=['linear', 'poly', 'rbf', 'sigmoid'], 
                        help='SVR核函數類型 (傳給分析腳本)')
    parser.add_argument('--svr_c', type=float, default=1.0, 
                        help='SVR正則化參數 (傳給分析腳本)')
    parser.add_argument('--svr_epsilon', type=float, default=0.1, 
                        help='SVR epsilon參數 (傳給分析腳本)')
    
    # 視覺化參數
    parser.add_argument('--dpi', type=int, default=300, 
                        help='圖片解析度')
    
    return parser.parse_args()

def setup_output_dir(dir_path):
    """建立輸出目錄"""
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def run_analysis_script(args, iteration):
    """執行分析腳本並收集結果"""
    cmd = [
        'python', args.py_path,
        '--data_file', args.data_file,
        '--encoding', args.encoding,
        '--model_type', args.model_type,
        # 使用時間+迭代次數作為隨機種子，確保每次執行都不同
        '--random_state', str(int(time.time()) % 10000 + iteration)
    ]
    
    # 如果使用SVR模型，添加SVR特定參數
    if args.model_type in ['svr', 'all']:
        cmd.extend(['--svr_kernel', args.svr_kernel])
        cmd.extend(['--svr_c', str(args.svr_c)])
        cmd.extend(['--svr_epsilon', str(args.svr_epsilon)])
    
    try:
        print(f"執行第 {iteration+1}/{args.n_iterations} 次迭代...")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"執行分析腳本時發生錯誤: {e}")
        return False

def extract_r2_values(result_dir, min_num=10):
    """從results.txt文件中提取R²值"""
    rf_r2_values = []
    xgb_r2_values = []
    mlr_r2_values = []  # 新增：多元線性迴歸
    svr_r2_values = []  # 新增：支持向量迴歸
    folder_nums = []
    
    result_path = Path(result_dir)
    result_folders = [f for f in result_path.glob('result_*') if f.is_dir()]
    
    # 正則表達式模式
    rf_pattern = r"隨機森林 R²:\s*([\d\.]+)"
    xgb_pattern = r"XGBoost R²:\s*([\d\.]+)"
    mlr_pattern = r"多元線性迴歸 R²:\s*([\d\.]+)"  # 新增模式
    svr_pattern = r"支持向量迴歸 R²:\s*([\d\.]+)"  # 新增模式
    
    for folder in result_folders:
        try:
            # 提取文件夾編號
            folder_num = int(folder.name.split('_')[1])
            
            # 僅處理編號大於min_num的文件夾
            if folder_num <= min_num:
                continue
                
            result_file = folder / 'results.txt'
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    has_data = False
                    
                    # 提取隨機森林R²
                    rf_match = re.search(rf_pattern, content)
                    if rf_match:
                        rf_r2 = float(rf_match.group(1))
                        rf_r2_values.append(rf_r2)
                        has_data = True
                    
                    # 提取XGBoostR²
                    xgb_match = re.search(xgb_pattern, content)
                    if xgb_match:
                        xgb_r2 = float(xgb_match.group(1))
                        xgb_r2_values.append(xgb_r2)
                        has_data = True
                    
                    # 提取多元線性迴歸R²
                    mlr_match = re.search(mlr_pattern, content)
                    if mlr_match:
                        mlr_r2 = float(mlr_match.group(1))
                        mlr_r2_values.append(mlr_r2)
                        has_data = True
                    
                    # 提取支持向量迴歸R²
                    svr_match = re.search(svr_pattern, content)
                    if svr_match:
                        svr_r2 = float(svr_match.group(1))
                        svr_r2_values.append(svr_r2)
                        has_data = True
                    
                    # 如果成功提取了任何值，記錄文件夾編號
                    if has_data:
                        folder_nums.append(folder_num)
        except Exception as e:
            print(f"處理文件夾 {folder} 時發生錯誤: {e}")
    # rf_r2_values = []
    # xgb_r2_values = []
    # mlr_r2_values = []  # 新增：多元線性迴歸
    # svr_r2_values = []  # 新增：支持向量迴歸
    return {
        'rf_r2': rf_r2_values,
        'xgb_r2': xgb_r2_values,
        'mlr_r2': mlr_r2_values,  # 新增
        'svr_r2': svr_r2_values,  # 新增
        'folder_nums': folder_nums
    }

def calculate_statistics(r2_values, ci_level=0.95):
    """計算R²的統計資料"""
    if not r2_values or len(r2_values) == 0:
        return None
    
    # 轉換為numpy數組
    r2_array = np.array(r2_values)
    
    # 計算基本統計量
    mean_r2 = np.mean(r2_array)
    median_r2 = np.median(r2_array)
    std_r2 = np.std(r2_array)
    min_r2 = np.min(r2_array)
    max_r2 = np.max(r2_array)
    
    # 計算信賴區間
    ci_lower = np.percentile(r2_array, (1 - ci_level) * 100 / 2)
    ci_upper = np.percentile(r2_array, 100 - (1 - ci_level) * 100 / 2)
    
    # 計算標準誤
    se = std_r2 / np.sqrt(len(r2_array))
    
    # 使用t分佈計算參數化信賴區間
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
    """視覺化R²分佈"""
    if not r2_values or len(r2_values) == 0:
        print(f"沒有足夠的{model_name} R²數據來繪製分佈圖")
        return
    
    plt.figure(figsize=(12, 7))
    
    # 設定中文字體顯示
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft JhengHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 繪製直方圖和KDE
    sns.histplot(r2_values, kde=True, color='steelblue')
    
    # 添加均值線
    plt.axvline(x=stats['mean'], color='red', linestyle='--', 
               label=f'平均值: {stats["mean"]:.4f}')
    
    # 添加中位數線
    plt.axvline(x=stats['median'], color='green', linestyle='-', 
               label=f'中位數: {stats["median"]:.4f}')
    
    # 添加信賴區間
    plt.axvline(x=stats['bootstrap_ci'][0], color='orange', linestyle=':', 
               label=f'Bootstrap {args.ci_level*100:.0f}% CI: [{stats["bootstrap_ci"][0]:.4f}, {stats["bootstrap_ci"][1]:.4f}]')
    plt.axvline(x=stats['bootstrap_ci'][1], color='orange', linestyle=':')
    
    # 圖標題和軸標籤
    plt.title(f'{model_name} R² 分佈 (n={stats["n_samples"]})', fontsize=22)
    plt.xlabel('R² 值', fontsize=18)
    plt.ylabel('頻率', fontsize=18)
    
    # 添加統計信息文本框
    stats_text = (
        f"基本統計量:\n"
        f"平均值: {stats['mean']:.6f}\n"
        f"中位數: {stats['median']:.6f}\n"
        f"標準差: {stats['std']:.6f}\n"
        f"最小值: {stats['min']:.6f}\n"
        f"最大值: {stats['max']:.6f}\n"
        f"標準誤: {stats['se']:.6f}\n"
        f"Bootstrap {args.ci_level*100:.0f}% CI: [{stats['bootstrap_ci'][0]:.6f}, {stats['bootstrap_ci'][1]:.6f}]\n"
        f"參數化 {args.ci_level*100:.0f}% CI: [{stats['parametric_ci'][0]:.6f}, {stats['parametric_ci'][1]:.6f}]"
    )
    plt.xlim(0.5, 1.0)   # 這行固定X軸
    plt.ylim(0, 80)  # 如果你要Y軸一致，這裡設置，否則可註解    
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                va='top', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()

    # 保存圖片
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_r2_distribution.png'), dpi=args.dpi)
    plt.close()

def visualize_r2_comparison(model_stats, output_dir, args):
    """視覺化所有模型的R²比較"""
    # 過濾出有效的模型統計
    valid_models = {name: stats for name, stats in model_stats.items() if stats is not None}
    
    if len(valid_models) < 2:
        print("沒有足夠的模型數據來繪製比較圖")
        return
    
    # ====== 新增：詳細統計信息輸出 ======
    print("\n" + "="*80)
    print("模型 R² 比較 (含信賴區間) - 詳細統計結果")
    print("="*80)
    
    for model_name, stats in valid_models.items():
        print(f"\n【{model_name}】")
        print(f"  樣本數量: {stats['n_samples']}")
        print(f"  平均 R²: {stats['mean']:.6f}")
        print(f"  中位數 R²: {stats['median']:.6f}")
        print(f"  標準差: {stats['std']:.6f}")
        print(f"  標準誤: {stats['se']:.6f}")
        print(f"  最小值: {stats['min']:.6f}")
        print(f"  最大值: {stats['max']:.6f}")
        print(f"  Bootstrap {args.ci_level*100:.0f}% 信賴區間: [{stats['bootstrap_ci'][0]:.6f}, {stats['bootstrap_ci'][1]:.6f}]")
        print(f"  信賴區間寬度: {stats['bootstrap_ci'][1] - stats['bootstrap_ci'][0]:.6f}")
        print(f"  參數化 {args.ci_level*100:.0f}% 信賴區間: [{stats['parametric_ci'][0]:.6f}, {stats['parametric_ci'][1]:.6f}]")
        print(f"  參數化CI寬度: {stats['parametric_ci'][1] - stats['parametric_ci'][0]:.6f}")
    
    # 模型間比較分析
    print(f"\n【模型間比較分析】")
    models = list(valid_models.keys())
    print(f"參與比較的模型: {', '.join(models)}")
    
    # 計算並顯示所有模型對之間的差異
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1 = models[i]
            model2 = models[j]
            mean_diff = valid_models[model1]['mean'] - valid_models[model2]['mean']
            
            # 檢查信賴區間是否重疊
            ci1_lower, ci1_upper = valid_models[model1]['bootstrap_ci']
            ci2_lower, ci2_upper = valid_models[model2]['bootstrap_ci']
            overlap = max(0, min(ci1_upper, ci2_upper) - max(ci1_lower, ci2_lower))
            ci_overlap = overlap > 0
            
            print(f"\n  {model1} vs {model2}:")
            print(f"    平均 R² 差異: {mean_diff:.6f}")
            print(f"    相對差異: {(mean_diff/valid_models[model2]['mean'])*100:.2f}%")
            print(f"    信賴區間重疊: {'是' if ci_overlap else '否'}")
            if ci_overlap:
                print(f"    重疊長度: {overlap:.6f}")
    
    # 排名分析
    print(f"\n【模型排名 (依平均 R² 值)】")
    sorted_models = sorted(valid_models.items(), key=lambda x: x[1]['mean'], reverse=True)
    for rank, (model_name, stats) in enumerate(sorted_models, 1):
        print(f"  第{rank}名: {model_name} (R² = {stats['mean']:.6f})")
    
    print("="*80)
    # ====== 原有的繪圖代碼 ======
    
    plt.figure(figsize=(12, 8))
    
    # 設定中文字體顯示
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft JhengHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 準備數據
    models = list(valid_models.keys())
    means = [valid_models[model]['mean'] for model in models]
    
    # 計算誤差條
    errors = []
    for model in models:
        stats = valid_models[model]
        lower_error = means[models.index(model)] - stats['bootstrap_ci'][0]
        upper_error = stats['bootstrap_ci'][1] - means[models.index(model)]
        errors.append([lower_error, upper_error])
    
    # 設置顏色映射
    model_colors = {
        '隨機森林': 'steelblue',
        'XGBoost': 'darkorange',
        '多元線性迴歸': 'forestgreen',
        '支持向量迴歸': 'firebrick'
    }
    
    colors = [model_colors.get(model, 'gray') for model in models]
    
    # 繪製條形圖
    bars = plt.bar(models, means, color=colors)
    
    # 添加誤差條
    plt.errorbar(models, means, yerr=np.array(errors).T, fmt='o', color='black', 
                ecolor='black', elinewidth=2, capsize=6)
    
    # 添加數值標籤
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{means[i]:.4f}', ha='center', va='bottom', fontsize=11)
    
    # 圖標題和軸標籤
    plt.title('模型 R² 比較 (含信賴區間)', fontsize=16)
    plt.ylabel('R² 值', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 設置y軸範圍，確保誤差條不會超出圖表
    all_ci_values = []
    for model in models:
        all_ci_values.extend(valid_models[model]['bootstrap_ci'])
    
    min_val = min(all_ci_values) - 0.05
    max_val = max(all_ci_values) + 0.05
    plt.ylim(min_val, max_val)
    
    plt.tight_layout()
    
    # 保存圖片
    plt.savefig(os.path.join(output_dir, 'model_r2_comparison.png'), dpi=args.dpi)
    plt.close()

def create_compatible_boxplot(data, labels=None, **kwargs):
    """
    創建兼容不同matplotlib版本的boxplot函數
    修復 labels 參數警告問題
    """
    try:
        from packaging import version
        matplotlib_version = version.parse(matplotlib.__version__)
        
        # Matplotlib 3.9+ 使用 tick_labels 參數
        if matplotlib_version >= version.parse("3.9.0"):
            if labels is not None:
                kwargs['tick_labels'] = labels
            return plt.boxplot(data, **kwargs)
        else:
            # 舊版本使用 labels 參數
            if labels is not None:
                kwargs['labels'] = labels
            return plt.boxplot(data, **kwargs)
    except ImportError:
        # 如果沒有packaging庫，直接使用labels（會有警告但功能正常）
        if labels is not None:
            kwargs['labels'] = labels
        return plt.boxplot(data, **kwargs)

def visualize_r2_boxplot(r2_data, model_names, output_dir, args):
    """視覺化R²的箱線圖比較 (修復版)"""
    # 定義模型名稱到r2_data鍵的映射
    model_key_mapping = {
        '隨機森林': 'rf',
        'XGBoost': 'xgb',
        '多元線性迴歸': 'mlr',
        '支持向量迴歸': 'svr'
    }
    
    # 過濾有效的模型數據
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
        print("沒有足夠的數據來繪製箱線圖")
        return
    
    # ====== 新增：詳細箱線圖統計信息輸出 ======
    print("\n" + "="*80)
    print("模型 R² 分佈比較 (箱線圖分析) - 詳細統計結果")
    print("="*80)
    
    for model in valid_labels:
        data = valid_data_dict[model]
        data_array = np.array(data)
        
        # 計算箱線圖統計量
        q1 = np.percentile(data_array, 25)
        q2 = np.percentile(data_array, 50)  # 中位數
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        
        # 計算異常值界限
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        # 找出異常值
        outliers = data_array[(data_array < lower_fence) | (data_array > upper_fence)]
        
        # 計算箱線圖的whiskers（鬍鬚）
        whisker_low = np.min(data_array[data_array >= lower_fence]) if len(data_array[data_array >= lower_fence]) > 0 else np.min(data_array)
        whisker_high = np.max(data_array[data_array <= upper_fence]) if len(data_array[data_array <= upper_fence]) > 0 else np.max(data_array)
        
        print(f"\n【{model}】(樣本數: {len(data)})")
        print(f"  五數摘要:")
        print(f"    最小值: {np.min(data_array):.6f}")
        print(f"    第一四分位數 (Q1): {q1:.6f}")
        print(f"    中位數 (Q2): {q2:.6f}")
        print(f"    第三四分位數 (Q3): {q3:.6f}")
        print(f"    最大值: {np.max(data_array):.6f}")
        print(f"  分佈特性:")
        print(f"    四分位距 (IQR): {iqr:.6f}")
        print(f"    下鬍鬚: {whisker_low:.6f}")
        print(f"    上鬍鬚: {whisker_high:.6f}")
        print(f"    數據範圍: {np.max(data_array) - np.min(data_array):.6f}")
        print(f"  異常值分析:")
        print(f"    異常值數量: {len(outliers)}")
        print(f"    異常值比例: {len(outliers)/len(data)*100:.2f}%")
        if len(outliers) > 0:
            print(f"    異常值: {', '.join([f'{x:.6f}' for x in sorted(outliers)])}")
        
        # 計算偏度和峰度
        from scipy import stats
        skewness = stats.skew(data_array)
        kurtosis = stats.kurtosis(data_array)
        print(f"  分佈形狀:")
        print(f"    偏度 (Skewness): {skewness:.4f} ({'右偏' if skewness > 0 else '左偏' if skewness < 0 else '對稱'})")
        print(f"    峰度 (Kurtosis): {kurtosis:.4f} ({'高峰' if kurtosis > 0 else '低峰' if kurtosis < 0 else '正常峰'})")
    
    # 模型間分佈比較
    print(f"\n【模型間分佈比較】")
    for i in range(len(valid_labels)):
        for j in range(i+1, len(valid_labels)):
            model1, model2 = valid_labels[i], valid_labels[j]
            data1, data2 = valid_data_dict[model1], valid_data_dict[model2]
            
            # 中位數差異
            median_diff = np.median(data1) - np.median(data2)
            
            # IQR比較
            iqr1 = np.percentile(data1, 75) - np.percentile(data1, 25)
            iqr2 = np.percentile(data2, 75) - np.percentile(data2, 25)
            iqr_ratio = iqr1 / iqr2 if iqr2 != 0 else float('inf')
            
            print(f"\n  {model1} vs {model2}:")
            print(f"    中位數差異: {median_diff:.6f}")
            print(f"    IQR比率: {iqr_ratio:.3f} ({model1}變異性{'較大' if iqr_ratio > 1 else '較小'})")
            
            # 進行Mann-Whitney U檢定（非參數檢定）
            try:
                u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                print(f"    Mann-Whitney U檢定: U={u_stat:.2f}, p={p_value:.6f}")
                print(f"    結論: {'有' if p_value < 0.05 else '沒有'}顯著分佈差異 (α=0.05)")
            except Exception as e:
                print(f"    Mann-Whitney U檢定失敗: {e}")
    
    # 分佈一致性分析
    print(f"\n【分佈穩定性排名】")
    stability_scores = []
    for model in valid_labels:
        data = valid_data_dict[model]
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        cv = np.std(data) / np.mean(data)  # 變異係數
        stability_score = 1 / (iqr + cv)  # 穩定性分數（越高越穩定）
        stability_scores.append((model, stability_score, iqr, cv))
    
    stability_scores.sort(key=lambda x: x[1], reverse=True)
    for rank, (model, score, iqr, cv) in enumerate(stability_scores, 1):
        print(f"  第{rank}名: {model} (IQR={iqr:.6f}, CV={cv:.4f})")
    
    print("="*80)
    # ====== 修復後的繪圖代碼 ======
    
    plt.figure(figsize=(10, 7))
    
    # 設定中文字體顯示
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft JhengHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 設置顏色映射
    model_colors = {
        '隨機森林': 'steelblue',
        'XGBoost': 'darkorange',
        '多元線性迴歸': 'forestgreen',
        '支持向量迴歸': 'firebrick'
    }
    
    # 使用修復後的兼容性boxplot函數
    box = create_compatible_boxplot(valid_data, labels=valid_labels, patch_artist=True)
    
    # 設置箱體顏色
    colors = [model_colors.get(model, 'gray') for model in valid_labels]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # 圖標題和軸標籤
    plt.title('模型 R² 分佈比較', fontsize=16)
    plt.ylabel('R² 值', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存圖片
    plt.savefig(os.path.join(output_dir, 'model_r2_boxplot.png'), dpi=args.dpi)
    plt.close()

def visualize_r2_trend(folder_nums, r2_data, model_names, output_dir, args):
    """視覺化R²隨迭代次數的趨勢"""
    if not folder_nums or len(folder_nums) == 0:
        print("沒有足夠的數據來繪製趨勢圖")
        return
    
    # 定義模型名稱到r2_data鍵的映射
    model_key_mapping = {
        '隨機森林': 'rf',
        'XGBoost': 'xgb',
        '多元線性迴歸': 'mlr',
        '支持向量迴歸': 'svr'
    }
    
    # 設置顏色和標記映射
    model_colors = {
        '隨機森林': ('steelblue', 'o-', 'navy'),
        'XGBoost': ('darkorange', 's-', 'darkred'),
        '多元線性迴歸': ('forestgreen', '^-', 'darkgreen'),
        '支持向量迴歸': ('firebrick', 'd-', 'maroon')
    }
    
    # 創建字典來對應文件夾編號與各個R²值
    model_data = {}
    for model in model_names:
        model_key = model_key_mapping.get(model)
        if model_key and f'{model_key}_r2' in r2_data and r2_data[f'{model_key}_r2']:
            model_data[model] = {}
            for i, folder in enumerate(folder_nums):
                if i < len(r2_data[f'{model_key}_r2']):
                    model_data[model][folder] = r2_data[f'{model_key}_r2'][i]
    
    # 檢查是否有有效數據
    if not model_data:
        print("沒有有效的模型數據來繪製趨勢圖")
        return
    
    # 排序文件夾編號
    sorted_folders = sorted(set(folder_nums))
    
    plt.figure(figsize=(14, 8))
    
    # 設定中文字體顯示
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft JhengHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 為每個模型繪製趨勢線
    for model in model_data:
        if not model_data[model]:
            continue
            
        # 準備繪圖數據
        model_folders = []
        model_values = []
        for folder in sorted_folders:
            if folder in model_data[model]:
                model_folders.append(folder)
                model_values.append(model_data[model][folder])
        
        if not model_folders:
            continue
            
        color, marker, ma_color = model_colors.get(model, ('gray', '*-', 'dimgray'))
        
        # 繪製R²趨勢
        plt.plot(model_folders, model_values, marker, color=color, label=f'{model} R²')
        
        # 繪製移動平均線（若點數足夠）
        window_size = min(5, len(model_folders) // 2) if len(model_folders) > 10 else 2
        if window_size > 1:
            model_ma = pd.Series(model_values).rolling(window=window_size).mean()
            plt.plot(model_folders, model_ma, '--', color=ma_color, 
                   label=f'{model} {window_size}點移動平均')
    
    # 圖標題和軸標籤
    plt.title('R² 隨迭代次數的變化趨勢', fontsize=16)
    plt.xlabel('執行次數', fontsize=14)
    plt.ylabel('R² 值', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # 保存圖片
    plt.savefig(os.path.join(output_dir, 'r2_trend_by_folder.png'), dpi=args.dpi)
    plt.close()

def save_statistics(model_stats, r2_data, output_dir, args):
    """保存統計結果"""
    # 定義模型名稱到r2_data鍵的映射
    model_key_mapping = {
        '隨機森林': 'rf',
        'XGBoost': 'xgb',
        '多元線性迴歸': 'mlr',
        '支持向量迴歸': 'svr'
    }
    
    with open(os.path.join(output_dir, 'bootstrap_statistics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Bootstrap 統計分析結果 (信賴水平: {args.ci_level*100}%)\n\n")
        
        # 保存各模型的統計結果
        for model_name, stats in model_stats.items():
            if stats:
                f.write(f"{model_name}模型 R² 統計 (樣本數: {stats['n_samples']}):\n")
                f.write(f"  平均值: {stats['mean']:.6f}\n")
                f.write(f"  中位數: {stats['median']:.6f}\n")
                f.write(f"  標準差: {stats['std']:.6f}\n")
                f.write(f"  標準誤: {stats['se']:.6f}\n")
                f.write(f"  最小值: {stats['min']:.6f}\n")
                f.write(f"  最大值: {stats['max']:.6f}\n")
                f.write(f"  Bootstrap {args.ci_level*100}% 信賴區間: [{stats['bootstrap_ci'][0]:.6f}, {stats['bootstrap_ci'][1]:.6f}]\n")
                f.write(f"  參數化 {args.ci_level*100}% 信賴區間: [{stats['parametric_ci'][0]:.6f}, {stats['parametric_ci'][1]:.6f}]\n")
                f.write("\n")
        
        # 保存模型比較
        valid_models = {name: stats for name, stats in model_stats.items() if stats is not None}
        if len(valid_models) >= 2:
            f.write("模型比較:\n")
            
            # 計算所有模型對之間的差異
            model_names = list(valid_models.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]
                    mean_diff = valid_models[model1]['mean'] - valid_models[model2]['mean']
                    f.write(f"  R² 平均值差異 ({model1} - {model2}): {mean_diff:.6f}\n")
            
            f.write("\n")
            
            # 嘗試進行t檢定
            try:
                # 為每個模型對建立字典，用於配對數據
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
                
                # 對每對模型進行t檢定
                for i in range(len(model_names)):
                    for j in range(i+1, len(model_names)):
                        model1 = model_names[i]
                        model2 = model_names[j]
                        
                        if model1 in model_dicts and model2 in model_dicts:
                            # 找出共同的文件夾
                            common_folders = set(model_dicts[model1].keys()) & set(model_dicts[model2].keys())
                            
                            if common_folders:
                                values1 = [model_dicts[model1][folder] for folder in common_folders]
                                values2 = [model_dicts[model2][folder] for folder in common_folders]
                                
                                # 使用這些配對數據進行t檢定
                                t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                                f.write(f"  {model1} vs {model2} 獨立樣本t檢定結果 (基於{len(common_folders)}個配對樣本):\n")
                                f.write(f"    t = {t_stat:.4f}, p = {p_value:.6f}\n")
                                f.write(f"    結論: {'有' if p_value < 0.05 else '沒有'}統計顯著性差異 (α = 0.05)\n\n")
                            else:
                                f.write(f"  {model1} vs {model2}: 無法進行t檢定 - 沒有足夠的配對數據\n\n")
            except Exception as e:
                f.write(f"  t檢定計算出錯: {e}\n\n")
    
    # 保存原始R²數據，方便後續分析
    try:
        # 創建DataFrame存儲所有模型的R²值
        result_data = []
        folder_nums_set = set(r2_data['folder_nums'])
        
        # 準備模型數據字典
        model_dicts = {}
        model_names = ['隨機森林', 'XGBoost', '多元線性迴歸', '支持向量迴歸']
        
        for model in model_names:
            model_key = model_key_mapping.get(model, model.lower().replace(" ", "_"))
            r2_key = f'{model_key}_r2'
            
            if r2_key in r2_data and r2_data[r2_key]:
                model_dicts[model] = {}
                for i, folder in enumerate(r2_data['folder_nums']):
                    if i < len(r2_data[r2_key]):
                        model_dicts[model][folder] = r2_data[r2_key][i]
        
        # 為每個文件夾創建一行數據
        for folder in sorted(folder_nums_set):
            row = {'folder_num': folder}
            for model in model_names:
                if model in model_dicts and folder in model_dicts[model]:
                    row[f'{model.lower().replace(" ", "_")}_r2'] = model_dicts[model][folder]
            result_data.append(row)
        
        # 創建DataFrame並保存
        pd.DataFrame(result_data).to_csv(os.path.join(output_dir, 'r2_values.csv'), index=False)
    except Exception as e:
        print(f"保存R²值CSV時發生錯誤: {e}")
        print("嘗試分別保存各個模型的R²值...")
        
        # 分別保存各模型的數據
        for model in model_names:
            model_key = model_key_mapping.get(model, model.lower().replace(" ", "_"))
            r2_key = f'{model_key}_r2'
            
            if r2_key in r2_data and r2_data[r2_key]:
                pd.DataFrame({
                    f'{model_key}_r2': r2_data[r2_key]
                }).to_csv(os.path.join(output_dir, f'{model_key}_r2_values.csv'), index=False)
                
def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 建立輸出目錄
    output_dir = setup_output_dir(args.output_dir)
    print(f"將分析結果儲存到: {output_dir}")
    
    # 檢查是否需要執行分析腳本
    need_to_run_analysis = args.n_iterations > 0
    
    if need_to_run_analysis:
        print(f"將執行 {args.n_iterations} 次 {args.py_path}...")
        for i in range(args.n_iterations):
            success = run_analysis_script(args, i)
            if not success:
                print(f"警告: 第 {i+1} 次迭代失敗")
    else:
        print(f"跳過執行 {args.py_path}，直接分析現有結果...")
    
    # 提取結果數據
    print(f"從 {args.result_dir} 中提取 R² 值，僅分析文件夾編號 > {args.min_num} 的結果...")
    r2_data = extract_r2_values(args.result_dir, args.min_num)
    
    # 檢查是否有足夠的數據進行分析
    model_keys = ['rf_r2', 'xgb_r2', 'mlr_r2', 'svr_r2']
    if not any(r2_data[key] for key in model_keys):
        print(f"錯誤: 未找到任何 R² 值。請檢查 {args.result_dir} 目錄和文件夾編號要求。")
        return
    
    # 顯示找到的數據量
    print(f"找到以下模型的R²值:")
    print(f"  隨機森林: {len(r2_data['rf_r2'])} 個")
    print(f"  XGBoost: {len(r2_data['xgb_r2'])} 個")
    print(f"  多元線性迴歸: {len(r2_data['mlr_r2'])} 個")
    print(f"  支持向量迴歸: {len(r2_data['svr_r2'])} 個")
    
    # 計算統計資料
    print(f"計算統計資料，使用 {args.ci_level*100}% 信賴水平...")
    model_stats = {
        '隨機森林': calculate_statistics(r2_data['rf_r2'], args.ci_level) if r2_data['rf_r2'] else None,
        'XGBoost': calculate_statistics(r2_data['xgb_r2'], args.ci_level) if r2_data['xgb_r2'] else None,
        '多元線性迴歸': calculate_statistics(r2_data['mlr_r2'], args.ci_level) if r2_data['mlr_r2'] else None,
        '支持向量迴歸': calculate_statistics(r2_data['svr_r2'], args.ci_level) if r2_data['svr_r2'] else None
    }
    
    # 視覺化 R² 分佈
    print("生成視覺化圖表...")
    # 定義模型名稱到r2_data鍵的映射
    model_key_mapping = {
        '隨機森林': 'rf',
        'XGBoost': 'xgb',
        '多元線性迴歸': 'mlr',
        '支持向量迴歸': 'svr'
    }

    for model_name, stats in model_stats.items():
        if stats:
            model_key = model_key_mapping.get(model_name)
            if model_key and f'{model_key}_r2' in r2_data:
                visualize_r2_distribution(r2_data[f'{model_key}_r2'], stats, model_name, output_dir, args)
            else:
                print(f"警告: 無法找到 {model_name} 的R²數據")
    
    # 視覺化模型比較
    valid_models = {name: stats for name, stats in model_stats.items() if stats is not None}
    if len(valid_models) >= 2:
        visualize_r2_comparison(valid_models, output_dir, args)
        visualize_r2_boxplot(r2_data, ['隨機森林', 'XGBoost', '多元線性迴歸', '支持向量迴歸'], output_dir, args)
    
    # 視覺化趨勢
    if len(r2_data['folder_nums']) > 1:
        visualize_r2_trend(r2_data['folder_nums'], r2_data, ['隨機森林', 'XGBoost', '多元線性迴歸', '支持向量迴歸'], output_dir, args)
    
    # 保存統計結果
    print("保存統計結果...")
    save_statistics(model_stats, r2_data, output_dir, args)
    
    print(f"\nBootstrap 分析完成！所有結果已保存到: {output_dir}")
    
    # 打印摘要統計
    for model_name, stats in model_stats.items():
        if stats:
            print(f"\n{model_name} R² 統計 (n={stats['n_samples']}):")
            print(f"  平均值: {stats['mean']:.4f}, {args.ci_level*100}% CI: [{stats['bootstrap_ci'][0]:.4f}, {stats['bootstrap_ci'][1]:.4f}]")
    
    # 打印模型比較
    valid_models = {name: stats for name, stats in model_stats.items() if stats is not None}
    if len(valid_models) >= 2:
        model_names = list(valid_models.keys())
        print(f"\n模型比較:")
        
        # 顯示所有模型對之間的差異
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                mean_diff = valid_models[model1]['mean'] - valid_models[model2]['mean']
                print(f"  R² 平均值差異 ({model1} - {model2}): {mean_diff:.4f}")
        
        # 嘗試進行t檢定分析
        try:
            # 為每個模型建立字典
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
            
            # 對每對模型進行t檢定
            print("\n模型間t檢定結果:")
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]
                    
                    if model1 in model_dicts and model2 in model_dicts:
                        # 找出共同的文件夾
                        common_folders = set(model_dicts[model1].keys()) & set(model_dicts[model2].keys())
                        
                        if common_folders:
                            values1 = [model_dicts[model1][folder] for folder in common_folders]
                            values2 = [model_dicts[model2][folder] for folder in common_folders]
                            
                            # 使用這些配對數據進行t檢定
                            t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                            print(f"  {model1} vs {model2} (基於{len(common_folders)}個樣本):")
                            print(f"    t={t_stat:.4f}, p={p_value:.4f}")
                            print(f"    結論: {'有' if p_value < 0.05 else '沒有'}統計顯著性差異 (α=0.05)")
                        else:
                            print(f"  {model1} vs {model2}: 無法進行t檢定 - 沒有足夠的配對數據")
        except Exception as e:
            print(f"  t檢定計算出錯: {e}")

if __name__ == "__main__":
    main()