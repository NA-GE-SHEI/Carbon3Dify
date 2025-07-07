# python enhanced_rf_analysis.py --data_file augmented_pro.csv --is_square 0 --is_round 1 --seat_area 707 --seat_thickness 3 --true_weight 4.2 --encoding utf-8
import pandas as pd
import numpy as np
import os, time
import argparse
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import seaborn as sns
from scipy import stats

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='椅子重量預測與分析工具 - 增強版隨機森林分析')
    
    # 主要參數
    parser.add_argument('--data_file', type=str, default='chair_data_V1.csv', help='資料檔案路徑')
    parser.add_argument('--encoding', type=str, default='big5', help='CSV檔案編碼')
    parser.add_argument('--output_dir', type=str, default='./enhanced_rf_result', help='輸出目錄')
    parser.add_argument('--model_type', type=str, default='all', 
                        choices=['rf', 'xgb', 'mlr', 'svr', 'all'], 
                        help='模型類型: rf (隨機森林), xgb (XGBoost), mlr (多元線性迴歸), svr (支持向量迴歸), all (全部)')
    
    parser.add_argument('--true_weight', type=float, default=4.2, help='椅子的真實重量 (kg)，如果不知道可以不設定')
    parser.add_argument('--is_square', type=int, default=0, help='是否為方形椅 (0=否, 1=是)')
    parser.add_argument('--is_round', type=int, default=1, help='是否為圓形椅 (0=否, 1=是)')
    parser.add_argument('--back_height', type=float, default=None, help='椅背高度 (cm), 可為None')
    parser.add_argument('--back_volume', type=float, default=0.000390 * 1000000, help='椅背體積 (cm3)')
    parser.add_argument('--seat_area', type=float, default=707, help='椅墊面積 (cm2)')
    parser.add_argument('--seat_thickness', type=float, default=3, help='椅墊厚度 (cm)')
    parser.add_argument('--leg_height', type=float, default=0.3737 * 100, help='椅腳高度 (cm)')
    parser.add_argument('--leg_volume', type=float, default=0.001550 * 1000000, help='椅腳體積 (cm3)')
    parser.add_argument('--seat_volume', type=float, default=0.002771 * 1000000, help='椅墊體積 (cm3)，用於計算')
    
    # 模型參數
    parser.add_argument('--test_size', type=float, default=0.2, help='測試集比例')
    parser.add_argument('--random_state', type=int, default=int(time.time()) % 10000, help='隨機種子 (默認使用當前時間)')
    parser.add_argument('--n_estimators', type=int, default=100, help='樹的數量')
    parser.add_argument('--max_depth', type=int, default=10, help='樹的最大深度')
    
    # SVR特定參數
    parser.add_argument('--svr_kernel', type=str, default='rbf', 
                        choices=['linear', 'poly', 'rbf', 'sigmoid'], 
                        help='SVR核函數類型')
    parser.add_argument('--svr_c', type=float, default=1.0, 
                        help='SVR正則化參數')
    parser.add_argument('--svr_epsilon', type=float, default=0.1, 
                        help='SVR epsilon參數')
    
    # 視覺化參數
    parser.add_argument('--dpi', type=int, default=300, help='圖片解析度')
    parser.add_argument('--comparison_chairs', type=str, default=None, 
                        help='比較的椅子資料，格式為: "名稱1:重量1,名稱2:重量2,..."')
    
    return parser.parse_args()

def setup_output_dir(base_dir):
    """建立輸出目錄"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 找出現有的最大結果編號
    max_num = 0
    for item in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, item)) and item.startswith('result'):
            try:
                num = int(item[6:])  # 提取'result'後的數字
                max_num = max(max_num, num)
            except ValueError:
                pass
    
    # 創建新的結果目錄
    new_num = max_num + 1
    new_result_dir = os.path.join(base_dir, f'result_{new_num}')
    
    # 處理目錄已存在的情況
    try:
        os.makedirs(new_result_dir)
    except FileExistsError:
        # 如果目錄已存在，尋找一個不存在的目錄名稱
        while os.path.exists(new_result_dir):
            new_num += 1
            new_result_dir = os.path.join(base_dir, f'result_{new_num}')
        os.makedirs(new_result_dir)
    
    print(f"將結果存儲到: {new_result_dir}")
    return new_result_dir

def prepare_data(file_path, encoding):
    """讀取和準備資料"""
    # 設定中文字體顯示，避免亂碼警告
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft JhengHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 讀取CSV檔案
    df = pd.read_csv(file_path, encoding=encoding)
    
    # 顯示資料集的前幾行以檢查
    print("資料集的前5行：")
    print(df.head())
    
    # 檢查缺失值
    print("\n缺失值檢查：")
    print(df.isnull().sum())
    
    # 將布林值轉換為整數 - 修正：直接強制轉換為整數型態
    if '方形椅' in df.columns:
        # 先將缺失值填充為0，再轉換為整數
        df['方形椅'] = df['方形椅'].fillna(0).astype(int)
    if '圓形椅' in df.columns:
        # 先將缺失值填充為0，再轉換為整數
        df['圓形椅'] = df['圓形椅'].fillna(0).astype(int)
    
    # 只處理數值型欄位，跳過文字型欄位
    numeric_cols = ['椅背高度(cm)', '椅背體積(cm)', '椅墊面積(cm)', 
                    '椅墊厚度(cm)', '椅腳高度(cm)', '椅腳體積(cm)', '重量(kg)']
    
    for col in numeric_cols:
        if col in df.columns:
            try:
                # 先檢查是否有字串類型的數值
                if df[col].dtype == 'object':
                    # 移除逗號並轉換為浮點數
                    df[col] = df[col].str.replace(',', '').astype(float)
                else:
                    # 已經是數值類型的直接確保為浮點數
                    df[col] = df[col].astype(float)
            except Exception as e:
                print(f"處理欄位 '{col}' 時發生錯誤: {e}")
    
    # 再次檢查資料型態
    print("\n資料型態檢查：")
    print(df.dtypes)
    
    return df

def format_feature_names_for_display(feature_names):
    """格式化特徵名稱以正確顯示單位"""
    formatted_names = []
    
    for name in feature_names:
        if '椅墊面積' in name:
            formatted_names.append('椅墊面積(cm²)')
        elif '椅背體積' in name:
            formatted_names.append('椅背體積(cm³)')
        elif '椅腳體積' in name:
            formatted_names.append('椅腳體積(cm³)')
        elif '椅背高度' in name:
            formatted_names.append('椅背高度(cm)')
        elif '椅墊厚度' in name:
            formatted_names.append('椅墊厚度(cm)')
        elif '椅腳高度' in name:
            formatted_names.append('椅腳高度(cm)')
        elif '方形椅' in name:
            formatted_names.append('方形椅')
        elif '圓形椅' in name:
            formatted_names.append('圓形椅')
        else:
            # 如果是其他未識別的特徵，保持原名
            formatted_names.append(name)
    
    return formatted_names

def preprocess_data(df, test_size, random_state):
    """資料預處理與分割，調整為你的CSV格式，並將缺失值替換成0"""

    # 定義特徵和目標變數 - 確保列名和資料集匹配
    feature_cols = ['方形椅', '圓形椅', '椅背高度(cm)', '椅背體積(cm)',
                    '椅墊面積(cm)', '椅墊厚度(cm)', '椅腳高度(cm)', '椅腳體積(cm)']
    target_col = '重量(kg)'

    # 確保所有列都存在
    for col in feature_cols:
        if col not in df.columns:
            print(f"警告: 列 '{col}' 不存在於資料集中")
    
    # 只使用存在的列
    valid_feature_cols = [col for col in feature_cols if col in df.columns]
    X = df[valid_feature_cols].fillna(0)
    
    if target_col not in df.columns:
        print(f"錯誤: 目標列 '{target_col}' 不存在於資料集中")
        return None, None, None, None, None, None, None, None, None, None
    
    y = df[target_col].fillna(0)

    # 缺失值處理器
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=valid_feature_cols)

    # 資料集分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 標準化特徵
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, imputer, scaler

def train_and_evaluate_rf_enhanced(X_train_scaled, y_train, X_test_scaled, y_test, X, args):
    """訓練與評估隨機森林模型 - 增強版"""
    print("\n訓練隨機森林模型...")
    
    # 建立隨機森林模型
    rf_model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=args.random_state,
        oob_score=True  # 啟用袋外評分
    )
    
    # 訓練模型
    rf_model.fit(X_train_scaled, y_train)
    
    # 預測和評估
    y_pred = rf_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"隨機森林模型評估:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"袋外評分 (OOB Score): {rf_model.oob_score_:.4f}")
    
    return rf_model, y_pred, rmse, r2, mae

def visualize_rf_feature_sensitivity(model, X, scaler, output_dir, args):
    """隨機森林特徵敏感度分析"""
    try:
        print("繪製特徵敏感度分析圖...")
        
        # 獲取特徵重要性
        importance = model.feature_importances_
        feature_names = X.columns
        
        # 格式化特徵名稱以正確顯示單位
        formatted_feature_names = format_feature_names_for_display(feature_names)
        
        # 按重要性排序
        indices = np.argsort(importance)[::-1]
        sorted_feature_names = [formatted_feature_names[i] for i in indices]
        sorted_importance = importance[indices]
        
        # 創建子圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左圖：特徵重要性條形圖
        colors = cm.Blues(np.linspace(0.4, 0.8, len(sorted_importance)))
        bars = ax1.bar(range(len(sorted_importance)), sorted_importance, 
                      color=colors, alpha=0.8)
        ax1.set_xticks(range(len(sorted_importance)))
        ax1.set_xticklabels(sorted_feature_names, rotation=45, ha='right')
        ax1.set_title('特徵重要性排序', fontsize=16)
        ax1.set_xlabel('特徵', fontsize=14)
        ax1.set_ylabel('重要性', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 右圖：特徵重要性圓餅圖
        colors_pie = cm.Set3(np.linspace(0, 1, len(sorted_importance)))
        wedges, texts, autotexts = ax2.pie(sorted_importance, labels=sorted_feature_names, 
                                          colors=colors_pie, autopct='%1.1f%%', 
                                          startangle=90)
        ax2.set_title('特徵重要性分佈', fontsize=16)
        
        # 設置圓餅圖文字大小
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
            autotext.set_fontsize(10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rf_feature_sensitivity_analysis.png'), dpi=args.dpi)
        plt.close()
    except Exception as e:
        print(f"繪製特徵敏感度分析圖時發生錯誤: {e}")

def visualize_rf_learning_curve(X, y, output_dir, args):
    """隨機森林學習曲線"""
    try:
        print("繪製學習曲線...")
        
        # 建立模型
        rf_model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=args.random_state
        )
        
        # 計算學習曲線
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            rf_model, X, y, cv=5, train_sizes=train_sizes, 
            scoring='neg_mean_squared_error', random_state=args.random_state
        )
        
        # 轉換為正的RMSE
        train_rmse_mean = np.sqrt(-train_scores.mean(axis=1))
        train_rmse_std = np.sqrt(train_scores.std(axis=1))
        val_rmse_mean = np.sqrt(-val_scores.mean(axis=1))
        val_rmse_std = np.sqrt(val_scores.std(axis=1))
        
        plt.figure(figsize=(12, 8))
        
        # 繪製訓練集學習曲線
        plt.plot(train_sizes_abs, train_rmse_mean, 'o-', color='blue', 
                label='訓練集 RMSE', linewidth=2)
        plt.fill_between(train_sizes_abs, train_rmse_mean - train_rmse_std,
                        train_rmse_mean + train_rmse_std, alpha=0.1, color='blue')
        
        # 繪製驗證集學習曲線
        plt.plot(train_sizes_abs, val_rmse_mean, 'o-', color='red', 
                label='驗證集 RMSE', linewidth=2)
        plt.fill_between(train_sizes_abs, val_rmse_mean - val_rmse_std,
                        val_rmse_mean + val_rmse_std, alpha=0.1, color='red')
        
        plt.title('隨機森林學習曲線', fontsize=16)
        plt.xlabel('訓練樣本數量', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加最佳點標記
        best_val_idx = np.argmin(val_rmse_mean)
        plt.scatter(train_sizes_abs[best_val_idx], val_rmse_mean[best_val_idx], 
                   color='green', s=100, zorder=5, label=f'最佳點: {train_sizes_abs[best_val_idx]} 樣本')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rf_learning_curve.png'), dpi=args.dpi)
        plt.close()
        
        # 返回詳細數據供後續分析
        return (train_sizes_abs, train_scores, val_scores)
        
    except Exception as e:
        print(f"繪製學習曲線時發生錯誤: {e}")
        return None

def visualize_rf_residual_heatmap(y_test, y_pred, output_dir, args):
    """殘差分析熱圖 (替代混淆矩陣)"""
    try:
        print("繪製殘差分析熱圖...")
        
        # 計算殘差
        residuals = y_test - y_pred
        
        # 創建子圖
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 殘差 vs 預測值散點圖
        ax1.scatter(y_pred, residuals, alpha=0.6, color='steelblue')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('預測值', fontsize=12)
        ax1.set_ylabel('殘差 (實際值 - 預測值)', fontsize=12)
        ax1.set_title('殘差 vs 預測值', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 2. 殘差直方圖
        ax2.hist(residuals, bins=20, color='lightblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('殘差', fontsize=12)
        ax2.set_ylabel('頻率', fontsize=12)
        ax2.set_title('殘差分佈直方圖', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q圖 (檢驗殘差正態性)
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('殘差正態性Q-Q圖', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # 4. 實際值 vs 預測值 + 殘差顏色映射
        scatter = ax4.scatter(y_test, y_pred, c=np.abs(residuals), 
                            cmap='YlOrRd', alpha=0.7, s=50)
        ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'k--', lw=2, label='完美預測線')
        ax4.set_xlabel('實際值', fontsize=12)
        ax4.set_ylabel('預測值', fontsize=12)
        ax4.set_title('實際值 vs 預測值 (顏色表示殘差大小)', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 添加顏色條
        plt.colorbar(scatter, ax=ax4, label='殘差絕對值')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rf_residual_heatmap_analysis.png'), dpi=args.dpi)
        plt.close()
        
    except Exception as e:
        print(f"繪製殘差分析熱圖時發生錯誤: {e}")

def visualize_rf_multiple_trees(model, X, output_dir, args, n_trees=3):
    """視覺化隨機森林中的多個決策樹"""
    try:
        print(f"繪製隨機森林中的 {n_trees} 個決策樹...")
        
        # 格式化特徵名稱以正確顯示單位
        formatted_feature_names = format_feature_names_for_display(X.columns)
        
        fig, axes = plt.subplots(1, n_trees, figsize=(25, 8))
        
        for i in range(n_trees):
            tree.plot_tree(model.estimators_[i], 
                          feature_names=formatted_feature_names,
                          filled=True,
                          rounded=True,
                          max_depth=3,
                          ax=axes[i])
            axes[i].set_title(f'決策樹 {i+1}', fontsize=14)
        
        plt.suptitle('隨機森林中的決策樹示例', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rf_multiple_decision_trees.png'), dpi=args.dpi)
        plt.close()
        
    except Exception as e:
        print(f"繪製多個決策樹時發生錯誤: {e}")

def visualize_rf_oob_error(X_train_scaled, y_train, output_dir, args):
    """袋外誤差圖"""
    try:
        print("繪製袋外誤差圖...")
        
        # 測試不同的樹數量
        n_estimators_range = range(10, 201, 10)
        oob_errors = []
        train_errors = []
        
        for n_est in n_estimators_range:
            # 建立模型
            rf_temp = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=args.max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=args.random_state,
                oob_score=True
            )
            
            # 訓練模型
            rf_temp.fit(X_train_scaled, y_train)
            
            # 計算袋外誤差
            oob_error = 1 - rf_temp.oob_score_
            oob_errors.append(oob_error)
            
            # 計算訓練誤差
            train_pred = rf_temp.predict(X_train_scaled)
            train_error = 1 - r2_score(y_train, train_pred)
            train_errors.append(train_error)
        
        # 繪製圖表
        plt.figure(figsize=(12, 8))
        plt.plot(n_estimators_range, oob_errors, 'o-', color='red', 
                label='袋外誤差 (OOB Error)', linewidth=2, markersize=6)
        plt.plot(n_estimators_range, train_errors, 'o-', color='blue', 
                label='訓練誤差', linewidth=2, markersize=6)
        
        plt.title('隨機森林袋外誤差 vs 樹的數量', fontsize=16)
        plt.xlabel('樹的數量 (n_estimators)', fontsize=14)
        plt.ylabel('誤差率 (1 - R²)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 標記最佳樹數量
        min_oob_idx = np.argmin(oob_errors)
        plt.scatter(n_estimators_range[min_oob_idx], oob_errors[min_oob_idx], 
                   color='green', s=100, zorder=5)
        plt.annotate(f'最佳樹數量: {n_estimators_range[min_oob_idx]}', 
                    xy=(n_estimators_range[min_oob_idx], oob_errors[min_oob_idx]),
                    xytext=(n_estimators_range[min_oob_idx]+20, oob_errors[min_oob_idx]+0.01),
                    fontsize=12, ha='left',
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rf_oob_error_analysis.png'), dpi=args.dpi)
        plt.close()
        
        # 返回詳細數據和最佳樹數量
        return n_estimators_range[min_oob_idx], (n_estimators_range, oob_errors, train_errors)
        
    except Exception as e:
        print(f"繪製袋外誤差圖時發生錯誤: {e}")
        return args.n_estimators, None

def visualize_rf_validation_curve(X, y, output_dir, args):
    """隨機森林驗證曲線 (超參數調優)"""
    try:
        print("繪製驗證曲線...")
        
        # 測試不同的max_depth值
        param_range = range(1, 21)
        
        train_scores, val_scores = validation_curve(
            RandomForestRegressor(n_estimators=50, random_state=args.random_state),
            X, y, param_name='max_depth', param_range=param_range,
            cv=5, scoring='neg_mean_squared_error'
        )
        
        # 轉換為正的RMSE
        train_rmse_mean = np.sqrt(-train_scores.mean(axis=1))
        train_rmse_std = np.sqrt(train_scores.std(axis=1))
        val_rmse_mean = np.sqrt(-val_scores.mean(axis=1))
        val_rmse_std = np.sqrt(val_scores.std(axis=1))
        
        plt.figure(figsize=(12, 8))
        
        # 繪製訓練分數
        plt.plot(param_range, train_rmse_mean, 'o-', color='blue', 
                label='訓練集 RMSE', linewidth=2)
        plt.fill_between(param_range, train_rmse_mean - train_rmse_std,
                        train_rmse_mean + train_rmse_std, alpha=0.1, color='blue')
        
        # 繪製驗證分數
        plt.plot(param_range, val_rmse_mean, 'o-', color='red', 
                label='驗證集 RMSE', linewidth=2)
        plt.fill_between(param_range, val_rmse_mean - val_rmse_std,
                        val_rmse_mean + val_rmse_std, alpha=0.1, color='red')
        
        plt.title('隨機森林驗證曲線 (max_depth)', fontsize=16)
        plt.xlabel('最大深度 (max_depth)', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 標記最佳參數
        best_depth_idx = np.argmin(val_rmse_mean)
        plt.scatter(param_range[best_depth_idx], val_rmse_mean[best_depth_idx], 
                   color='green', s=100, zorder=5)
        plt.annotate(f'最佳深度: {param_range[best_depth_idx]}', 
                    xy=(param_range[best_depth_idx], val_rmse_mean[best_depth_idx]),
                    xytext=(param_range[best_depth_idx]+2, val_rmse_mean[best_depth_idx]+0.1),
                    fontsize=12, ha='left',
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rf_validation_curve.png'), dpi=args.dpi)
        plt.close()
        
        # 返回詳細數據和最佳深度
        return param_range[best_depth_idx], (param_range, train_scores, val_scores)
        
    except Exception as e:
        print(f"繪製驗證曲線時發生錯誤: {e}")
        return args.max_depth, None

def create_new_chair_data(args):
    """建立新椅子的特徵資料，符合您的CSV欄位格式，並將缺失值替換成0"""
    # 修正：確保欄位名稱與資料集中的列名完全一致
    new_chair = pd.DataFrame({
        '方形椅': [args.is_square],
        '圓形椅': [args.is_round],
        '椅背高度(cm)': [args.back_height if args.back_height is not None else 0],
        '椅背體積(cm)': [args.back_volume if args.back_volume is not None else 0],
        '椅墊面積(cm)': [args.seat_area if args.seat_area is not None else 0],
        '椅墊厚度(cm)': [args.seat_thickness if args.seat_thickness is not None else 0],
        '椅腳高度(cm)': [args.leg_height if args.leg_height is not None else 0],
        '椅腳體積(cm)': [args.leg_volume if args.leg_volume is not None else 0]
    })

    return new_chair

def save_detailed_analysis_results(model, y_test, y_pred, best_n_estimators, best_max_depth, feature_names, 
                                  learning_curve_data, oob_data, validation_curve_data, predicted_weight, args, output_dir):
    """保存詳細的分析結果到txt文件"""
    try:
        # 計算各種評估指標
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        residuals = y_test - y_pred
        
        with open(os.path.join(output_dir, 'detailed_analysis_results.txt'), 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("隨機森林增強分析詳細結果報告\n")
            f.write("=" * 80 + "\n\n")
            
            # 基本模型信息
            f.write("【模型基本信息】\n")
            f.write("-" * 40 + "\n")
            f.write(f"資料文件: {args.data_file}\n")
            f.write(f"編碼格式: {args.encoding}\n")
            f.write(f"測試集比例: {args.test_size}\n")
            f.write(f"隨機種子: {args.random_state}\n")
            f.write(f"訓練樣本數: {len(y_test) / args.test_size * (1 - args.test_size):.0f}\n")
            f.write(f"測試樣本數: {len(y_test)}\n")
            f.write("\n")
            
            # 模型評估指標
            f.write("【模型評估指標】\n")
            f.write("-" * 40 + "\n")
            f.write(f"決定係數 (R²): {r2:.8f}\n")
            f.write(f"均方根誤差 (RMSE): {rmse:.8f}\n")
            f.write(f"平均絕對誤差 (MAE): {mae:.8f}\n")
            f.write(f"袋外評分 (OOB Score): {model.oob_score_:.8f}\n")
            f.write(f"模型解釋變異度: {r2 * 100:.4f}%\n")
            f.write("\n")
            
            # 新椅子預測結果詳細信息
            f.write("【新椅子預測結果】\n")
            f.write("-" * 40 + "\n")
            if predicted_weight is not None:
                f.write(f"預測重量: {predicted_weight:.2f} kg\n")
                
                if args.true_weight is not None:
                    error = abs(predicted_weight - args.true_weight)
                    relative_error = error / args.true_weight * 100
                    f.write(f"真實重量: {args.true_weight:.2f} kg\n")
                    f.write(f"絕對誤差: {error:.2f} kg\n")
                    f.write(f"相對誤差: {relative_error:.2f}%\n")
                    
                    # 預測準確度詳細分析
                    if relative_error <= 5:
                        accuracy_grade = "優秀"
                        accuracy_desc = "預測非常準確，誤差在可接受範圍內"
                    elif relative_error <= 10:
                        accuracy_grade = "良好"
                        accuracy_desc = "預測較為準確，誤差在合理範圍內"
                    elif relative_error <= 15:
                        accuracy_grade = "一般"
                        accuracy_desc = "預測基本準確，有一定改進空間"
                    else:
                        accuracy_grade = "需改進"
                        accuracy_desc = "預測誤差較大，建議檢查模型或特徵"
                    
                    f.write(f"預測準確度評級: {accuracy_grade}\n")
                    f.write(f"評級說明: {accuracy_desc}\n")
                    
                    # 與模型整體性能比較
                    model_avg_relative_error = np.mean(np.abs(residuals) / np.abs(y_test) * 100)
                    if relative_error <= model_avg_relative_error:
                        performance_comparison = "優於模型平均水平"
                    else:
                        performance_comparison = "低於模型平均水平"
                    f.write(f"與模型平均性能比較: {performance_comparison} (模型平均相對誤差: {model_avg_relative_error:.2f}%)\n")
                else:
                    f.write("真實重量: 未提供\n")
                    f.write("預測誤差: 無法計算 (未提供真實重量)\n")
                    f.write("建議: 如獲得實際重量，可重新運行分析進行誤差評估\n")
            else:
                f.write("預測狀態: 失敗\n")
                f.write("建議: 檢查輸入特徵數據格式和完整性\n")
            
            # 椅子特徵參數詳細信息
            f.write(f"\n椅子特徵參數:\n")
            f.write(f"  是否為方形椅: {'是' if args.is_square else '否'}\n")
            f.write(f"  是否為圓形椅: {'是' if args.is_round else '否'}\n")
            f.write(f"  椅背高度(cm): {args.back_height if args.back_height is not None else '未設定'}\n")
            f.write(f"  椅背體積(cm³): {args.back_volume:.2f}\n")
            f.write(f"  椅墊面積(cm²): {args.seat_area:.2f}\n")
            f.write(f"  椅墊厚度(cm): {args.seat_thickness:.2f}\n")
            f.write(f"  椅腳高度(cm): {args.leg_height:.2f}\n")
            f.write(f"  椅腳體積(cm³): {args.leg_volume:.2f}\n")
            f.write("\n")
            
            # 詳細殘差統計
            f.write("【殘差詳細統計】\n")
            f.write("-" * 40 + "\n")
            f.write(f"殘差平均值: {np.mean(residuals):.8f}\n")
            f.write(f"殘差中位數: {np.median(residuals):.8f}\n")
            f.write(f"殘差標準差: {np.std(residuals):.8f}\n")
            f.write(f"殘差變異數: {np.var(residuals):.8f}\n")
            f.write(f"殘差最小值: {np.min(residuals):.8f}\n")
            f.write(f"殘差最大值: {np.max(residuals):.8f}\n")
            f.write(f"殘差範圍: {np.max(residuals) - np.min(residuals):.8f}\n")
            f.write(f"殘差四分位距 (IQR): {np.percentile(residuals, 75) - np.percentile(residuals, 25):.8f}\n")
            f.write(f"殘差偏度 (Skewness): {stats.skew(residuals):.8f}\n")
            f.write(f"殘差峰度 (Kurtosis): {stats.kurtosis(residuals):.8f}\n")
            
            # 殘差百分位數
            f.write("\n殘差百分位數分佈:\n")
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            for p in percentiles:
                f.write(f"  {p:2d}%: {np.percentile(residuals, p):.8f}\n")
            f.write("\n")
            
            # 預測精度分析
            f.write("【預測精度分析】\n")
            f.write("-" * 40 + "\n")
            abs_errors = np.abs(residuals)
            relative_errors = abs_errors / np.abs(y_test) * 100
            
            f.write(f"平均絕對誤差: {np.mean(abs_errors):.8f}\n")
            f.write(f"絕對誤差標準差: {np.std(abs_errors):.8f}\n")
            f.write(f"平均相對誤差: {np.mean(relative_errors):.4f}%\n")
            f.write(f"相對誤差標準差: {np.std(relative_errors):.4f}%\n")
            f.write(f"最大絕對誤差: {np.max(abs_errors):.8f}\n")
            f.write(f"最大相對誤差: {np.max(relative_errors):.4f}%\n")
            
            # 精度等級統計
            accuracy_levels = [0.05, 0.1, 0.15, 0.2, 0.3]
            f.write("\n不同精度等級下的預測準確率:\n")
            for level in accuracy_levels:
                accurate_predictions = np.sum(relative_errors <= level * 100)
                accuracy_rate = accurate_predictions / len(relative_errors) * 100
                f.write(f"  誤差 ≤ {level*100:4.1f}%: {accurate_predictions:4d}/{len(relative_errors):4d} ({accuracy_rate:5.1f}%)\n")
            f.write("\n")
            
            # 模型優化結果
            f.write("【模型優化結果】\n")
            f.write("-" * 40 + "\n")
            f.write(f"初始樹數量: {args.n_estimators}\n")
            f.write(f"最佳樹數量: {best_n_estimators}\n")
            f.write(f"初始最大深度: {args.max_depth}\n")
            f.write(f"最佳最大深度: {best_max_depth}\n")
            f.write(f"樹數量改善: {((best_n_estimators - args.n_estimators) / args.n_estimators * 100):+.1f}%\n")
            f.write(f"深度改善: {((best_max_depth - args.max_depth) / args.max_depth * 100):+.1f}%\n")
            f.write("\n")
            
            # 特徵重要性詳細分析
            f.write("【特徵重要性詳細分析】\n")
            f.write("-" * 40 + "\n")
            
            # 使用傳入的特徵名稱，兼容不同版本的scikit-learn
            if hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_
            else:
                features = feature_names
            
            # 格式化特徵名稱以正確顯示單位
            formatted_features = format_feature_names_for_display(features)
            
            feature_importance = list(zip(formatted_features, model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            f.write("特徵重要性排序:\n")
            total_importance = sum([imp for _, imp in feature_importance])
            cumulative_importance = 0
            
            for i, (feature, importance) in enumerate(feature_importance, 1):
                relative_importance = importance / total_importance * 100
                cumulative_importance += relative_importance
                f.write(f"  {i:2d}. {feature:<20}: {importance:.8f} ({relative_importance:5.2f}%, 累計: {cumulative_importance:5.2f}%)\n")
            
            # 特徵重要性統計
            importances = [imp for _, imp in feature_importance]
            f.write(f"\n特徵重要性統計:\n")
            f.write(f"  平均重要性: {np.mean(importances):.8f}\n")
            f.write(f"  重要性標準差: {np.std(importances):.8f}\n")
            f.write(f"  最高重要性: {np.max(importances):.8f}\n")
            f.write(f"  最低重要性: {np.min(importances):.8f}\n")
            f.write(f"  重要性比值 (最高/最低): {np.max(importances)/np.min(importances):.2f}\n")
            f.write("\n")
            
            # 學習曲線詳細數據
            if learning_curve_data:
                f.write("【學習曲線詳細數據】\n")
                f.write("-" * 40 + "\n")
                train_sizes_abs, train_scores, val_scores = learning_curve_data
                
                f.write("訓練樣本數量 | 訓練RMSE (平均±標準差) | 驗證RMSE (平均±標準差)\n")
                f.write("-" * 70 + "\n")
                
                for i, size in enumerate(train_sizes_abs):
                    train_rmse_mean = np.sqrt(-train_scores[i].mean())
                    train_rmse_std = np.sqrt(train_scores[i].std())
                    val_rmse_mean = np.sqrt(-val_scores[i].mean())
                    val_rmse_std = np.sqrt(val_scores[i].std())
                    
                    f.write(f"{size:8.0f}     | {train_rmse_mean:8.6f}±{train_rmse_std:8.6f} | {val_rmse_mean:8.6f}±{val_rmse_std:8.6f}\n")
                f.write("\n")
            
            # 袋外誤差詳細數據
            if oob_data:
                f.write("【袋外誤差詳細數據】\n")
                f.write("-" * 40 + "\n")
                n_estimators_range, oob_errors, train_errors = oob_data
                
                f.write("樹數量 | 袋外誤差 | 訓練誤差 | 誤差差異\n")
                f.write("-" * 45 + "\n")
                
                for i, n_est in enumerate(n_estimators_range):
                    error_diff = abs(oob_errors[i] - train_errors[i])
                    f.write(f"{n_est:6d} | {oob_errors[i]:8.6f} | {train_errors[i]:8.6f} | {error_diff:8.6f}\n")
                
                min_oob_idx = np.argmin(oob_errors)
                f.write(f"\n最佳樹數量: {n_estimators_range[min_oob_idx]} (袋外誤差: {oob_errors[min_oob_idx]:.6f})\n")
                f.write("\n")
            
            # 驗證曲線詳細數據
            if validation_curve_data:
                f.write("【驗證曲線詳細數據】\n")
                f.write("-" * 40 + "\n")
                param_range, train_scores, val_scores = validation_curve_data
                
                f.write("最大深度 | 訓練RMSE (平均±標準差) | 驗證RMSE (平均±標準差)\n")
                f.write("-" * 65 + "\n")
                
                for i, depth in enumerate(param_range):
                    train_rmse_mean = np.sqrt(-train_scores[i].mean())
                    train_rmse_std = np.sqrt(train_scores[i].std())
                    val_rmse_mean = np.sqrt(-val_scores[i].mean())
                    val_rmse_std = np.sqrt(val_scores[i].std())
                    
                    f.write(f"{depth:6d}     | {train_rmse_mean:8.6f}±{train_rmse_std:8.6f} | {val_rmse_mean:8.6f}±{val_rmse_std:8.6f}\n")
                
                best_depth_idx = np.argmin(np.sqrt(-val_scores.mean(axis=1)))
                f.write(f"\n最佳深度: {param_range[best_depth_idx]} (驗證RMSE: {np.sqrt(-val_scores[best_depth_idx].mean()):.6f})\n")
                f.write("\n")
            
            # 預測結果詳細統計
            f.write("【預測結果詳細統計】\n")
            f.write("-" * 40 + "\n")
            f.write(f"實際值統計:\n")
            f.write(f"  平均值: {np.mean(y_test):.8f}\n")
            f.write(f"  標準差: {np.std(y_test):.8f}\n")
            f.write(f"  最小值: {np.min(y_test):.8f}\n")
            f.write(f"  最大值: {np.max(y_test):.8f}\n")
            f.write(f"  範圍: {np.max(y_test) - np.min(y_test):.8f}\n")
            
            f.write(f"\n預測值統計:\n")
            f.write(f"  平均值: {np.mean(y_pred):.8f}\n")
            f.write(f"  標準差: {np.std(y_pred):.8f}\n")
            f.write(f"  最小值: {np.min(y_pred):.8f}\n")
            f.write(f"  最大值: {np.max(y_pred):.8f}\n")
            f.write(f"  範圍: {np.max(y_pred) - np.min(y_pred):.8f}\n")
            
            # 相關性分析
            correlation = np.corrcoef(y_test, y_pred)[0, 1]
            f.write(f"\n實際值與預測值相關係數: {correlation:.8f}\n")
            f.write("\n")
            
            # 模型參數總結
            f.write("【模型參數總結】\n")
            f.write("-" * 40 + "\n")
            f.write(f"樹的數量 (n_estimators): {model.n_estimators}\n")
            f.write(f"最大深度 (max_depth): {model.max_depth}\n")
            f.write(f"最小分割樣本數 (min_samples_split): {model.min_samples_split}\n")
            f.write(f"最小葉節點樣本數 (min_samples_leaf): {model.min_samples_leaf}\n")
            f.write(f"最大特徵數 (max_features): {model.max_features}\n")
            f.write(f"隨機種子: {model.random_state}\n")
            f.write(f"啟用袋外評分: {model.oob_score is not None}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("報告生成完成\n")
            f.write("=" * 80 + "\n")
            
    except Exception as e:
        print(f"保存詳細分析結果時發生錯誤: {e}")
            
    except Exception as e:
        print(f"保存詳細分析結果時發生錯誤: {e}")

def save_raw_data_for_analysis(y_test, y_pred, learning_curve_data, oob_data, validation_curve_data, 
                              predicted_weight, args, output_dir):
    """保存原始數據到CSV文件供進一步分析"""
    try:
        print("保存原始數據到CSV文件...")
        
        # 1. 保存預測結果
        prediction_df = pd.DataFrame({
            'actual_values(kg)': y_test,
            'predicted_values(kg)': y_pred,
            'residuals(kg)': y_test - y_pred,
            'absolute_errors(kg)': np.abs(y_test - y_pred),
            'relative_errors(%)': np.abs(y_test - y_pred) / np.abs(y_test) * 100
        })
        prediction_df.to_csv(os.path.join(output_dir, 'prediction_results.csv'), index=False)
        
        # 2. 保存新椅子預測信息
        chair_prediction_data = {
            'predicted_weight(kg)': [predicted_weight if predicted_weight is not None else 'N/A'],
            'true_weight(kg)': [args.true_weight if args.true_weight is not None else 'N/A'],
            'absolute_error(kg)': [abs(predicted_weight - args.true_weight) if predicted_weight is not None and args.true_weight is not None else 'N/A'],
            'relative_error(%)': [abs(predicted_weight - args.true_weight) / args.true_weight * 100 if predicted_weight is not None and args.true_weight is not None else 'N/A'],
            'is_square': [args.is_square],
            'is_round': [args.is_round],
            'back_height(cm)': [args.back_height if args.back_height is not None else 'N/A'],
            'back_volume(cm³)': [args.back_volume],
            'seat_area(cm²)': [args.seat_area],
            'seat_thickness(cm)': [args.seat_thickness],
            'leg_height(cm)': [args.leg_height],
            'leg_volume(cm³)': [args.leg_volume]
        }
        chair_df = pd.DataFrame(chair_prediction_data)
        chair_df.to_csv(os.path.join(output_dir, 'new_chair_prediction.csv'), index=False)
        
        # 3. 保存學習曲線數據
        if learning_curve_data:
            train_sizes_abs, train_scores, val_scores = learning_curve_data
            learning_df = pd.DataFrame({
                'train_size': train_sizes_abs,
                'train_rmse_mean(kg)': np.sqrt(-train_scores.mean(axis=1)),
                'train_rmse_std(kg)': np.sqrt(train_scores.std(axis=1)),
                'val_rmse_mean(kg)': np.sqrt(-val_scores.mean(axis=1)),
                'val_rmse_std(kg)': np.sqrt(val_scores.std(axis=1))
            })
            learning_df.to_csv(os.path.join(output_dir, 'learning_curve_data.csv'), index=False)
        
        # 4. 保存袋外誤差數據
        if oob_data:
            n_estimators_range, oob_errors, train_errors = oob_data
            oob_df = pd.DataFrame({
                'n_estimators': n_estimators_range,
                'oob_error(1-R²)': oob_errors,
                'train_error(1-R²)': train_errors,
                'error_difference': np.abs(np.array(oob_errors) - np.array(train_errors))
            })
            oob_df.to_csv(os.path.join(output_dir, 'oob_error_data.csv'), index=False)
        
        # 5. 保存驗證曲線數據
        if validation_curve_data:
            param_range, train_scores, val_scores = validation_curve_data
            validation_df = pd.DataFrame({
                'max_depth': param_range,
                'train_rmse_mean(kg)': np.sqrt(-train_scores.mean(axis=1)),
                'train_rmse_std(kg)': np.sqrt(train_scores.std(axis=1)),
                'val_rmse_mean(kg)': np.sqrt(-val_scores.mean(axis=1)),
                'val_rmse_std(kg)': np.sqrt(val_scores.std(axis=1))
            })
            validation_df.to_csv(os.path.join(output_dir, 'validation_curve_data.csv'), index=False)
        
        print("原始數據已保存到CSV文件中，可用於進一步分析")
        
    except Exception as e:
        print(f"保存原始數據時發生錯誤: {e}")

def save_enhanced_results(model, y_test, y_pred, best_n_estimators, best_max_depth, feature_names, 
                         predicted_weight, args, output_dir):
    """保存增強分析結果 (簡化版)"""
    try:
        # 計算各種評估指標
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        residuals = y_test - y_pred
        
        with open(os.path.join(output_dir, 'enhanced_rf_results.txt'), 'w', encoding='utf-8') as f:
            f.write("隨機森林增強分析結果：\n\n")
            
            # 模型評估指標
            f.write("模型評估指標：\n")
            f.write(f"R²: {r2:.6f}\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"袋外評分 (OOB Score): {model.oob_score_:.6f}\n")
            f.write("\n")
            
            # 殘差統計
            f.write("殘差統計：\n")
            f.write(f"殘差平均值: {np.mean(residuals):.6f}\n")
            f.write(f"殘差標準差: {np.std(residuals):.6f}\n")
            f.write(f"殘差最小值: {np.min(residuals):.6f}\n")
            f.write(f"殘差最大值: {np.max(residuals):.6f}\n")
            f.write("\n")
            
            # 最佳超參數
            f.write("模型優化結果：\n")
            f.write(f"最佳樹數量: {best_n_estimators}\n")
            f.write(f"最佳最大深度: {best_max_depth}\n")
            f.write("\n")
            
            # 特徵重要性
            f.write("特徵重要性排序：\n")
            # 使用傳入的特徵名稱，兼容不同版本的scikit-learn
            if hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_
            else:
                features = feature_names
            
            # 格式化特徵名稱以正確顯示單位
            formatted_features = format_feature_names_for_display(features)
            
            feature_importance = list(zip(formatted_features, model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(feature_importance, 1):
                f.write(f"{i}. {feature}: {importance:.6f}\n")
            f.write("\n")
            
            # 新椅子預測結果
            f.write("新椅子預測結果：\n")
            if predicted_weight is not None:
                f.write(f"預測重量: {predicted_weight:.2f} kg\n")
                
                if args.true_weight is not None:
                    error = abs(predicted_weight - args.true_weight)
                    relative_error = error / args.true_weight * 100
                    f.write(f"真實重量: {args.true_weight:.2f} kg\n")
                    f.write(f"預測誤差: {error:.2f} kg ({relative_error:.2f}%)\n")
                    
                    # 預測準確度評級
                    if relative_error <= 5:
                        accuracy_grade = "優秀"
                    elif relative_error <= 10:
                        accuracy_grade = "良好"
                    elif relative_error <= 15:
                        accuracy_grade = "一般"
                    else:
                        accuracy_grade = "需改進"
                    
                    f.write(f"預測準確度評級: {accuracy_grade}\n")
                else:
                    f.write("真實重量: 未提供\n")
                    f.write("預測誤差: 無法計算 (未提供真實重量)\n")
            else:
                f.write("預測失敗\n")
            f.write("\n")
            
            # 椅子特徵參數
            f.write("椅子特徵參數：\n")
            f.write(f"是否為方形椅: {'是' if args.is_square else '否'}\n")
            f.write(f"是否為圓形椅: {'是' if args.is_round else '否'}\n")
            f.write(f"椅背高度(cm): {args.back_height if args.back_height is not None else '未設定'}\n")
            f.write(f"椅背體積(cm³): {args.back_volume:.2f}\n")
            f.write(f"椅墊面積(cm²): {args.seat_area:.2f}\n")
            f.write(f"椅墊厚度(cm): {args.seat_thickness:.2f}\n")
            f.write(f"椅腳高度(cm): {args.leg_height:.2f}\n")
            f.write(f"椅腳體積(cm³): {args.leg_volume:.2f}\n")
            
    except Exception as e:
        print(f"保存增強分析結果時發生錯誤: {e}")

def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 建立輸出目錄
    output_dir = setup_output_dir(args.output_dir)
    
    # 讀取和準備資料
    df = prepare_data(args.data_file, args.encoding)
    
    # 資料預處理與分割
    X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, imputer, scaler = preprocess_data(df, args.test_size, args.random_state)
    
    # 檢查資料是否正確處理
    if X is None or y is None:
        print("錯誤: 資料預處理失敗")
        return None
    
    print(f"\n開始增強版隨機森林分析...")
    
    # 1. 訓練基本隨機森林模型
    rf_model, rf_y_pred, rf_rmse, rf_r2, rf_mae = train_and_evaluate_rf_enhanced(
        X_train_scaled, y_train, X_test_scaled, y_test, X, args)
    
    # 2. 特徵敏感度分析 (特徵重要性圖)
    visualize_rf_feature_sensitivity(rf_model, X, scaler, output_dir, args)
    
    # 3. 學習曲線圖
    learning_curve_data = visualize_rf_learning_curve(X, y, output_dir, args)
    
    # 4. 殘差分析熱圖 (替代混淆矩陣，因為這是回歸問題)
    visualize_rf_residual_heatmap(y_test, rf_y_pred, output_dir, args)
    
    # 5. 多個決策樹視覺化
    visualize_rf_multiple_trees(rf_model, X, output_dir, args, n_trees=3)
    
    # 6. 袋外誤差圖
    best_n_estimators, oob_data = visualize_rf_oob_error(X_train_scaled, y_train, output_dir, args)
    
    # 7. 驗證曲線 (超參數調優)
    best_max_depth, validation_curve_data = visualize_rf_validation_curve(X, y, output_dir, args)
    
    # 8. 使用最佳參數重新訓練模型
    print(f"\n使用最佳參數重新訓練模型...")
    print(f"最佳樹數量: {best_n_estimators}")
    print(f"最佳最大深度: {best_max_depth}")
    
    optimized_rf_model = RandomForestRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=args.random_state,
        oob_score=True
    )
    
    optimized_rf_model.fit(X_train_scaled, y_train)
    optimized_y_pred = optimized_rf_model.predict(X_test_scaled)
    optimized_rmse = np.sqrt(mean_squared_error(y_test, optimized_y_pred))
    optimized_r2 = r2_score(y_test, optimized_y_pred)
    
    print(f"優化後模型評估:")
    print(f"RMSE: {optimized_rmse:.4f}")
    print(f"R²: {optimized_r2:.4f}")
    print(f"袋外評分: {optimized_rf_model.oob_score_:.4f}")
    
    # 9. 新椅子預測
    try:
        new_chair = create_new_chair_data(args)
        
        # 檢查欄位名稱是否一致
        for col in X.columns:
            if col not in new_chair.columns:
                print(f"警告: 新椅子資料中缺少欄位 '{col}'")
                new_chair[col] = 0
        
        # 確保欄位順序一致
        new_chair = new_chair[X.columns]
        
        # 處理缺失值並標準化
        new_chair_imputed = imputer.transform(new_chair)
        new_chair = pd.DataFrame(new_chair_imputed, columns=X.columns)
        new_chair_scaled = scaler.transform(new_chair)
        
        # 預測
        predicted_weight = optimized_rf_model.predict(new_chair_scaled)[0]
        print(f"\n新椅子的預測重量: {predicted_weight:.2f} kg")
        
        if args.true_weight is not None:
            error = abs(predicted_weight - args.true_weight)
            relative_error = error / args.true_weight * 100
            print(f"預測誤差: {error:.2f} kg ({relative_error:.2f}%)")
        
    except Exception as e:
        print(f"預測新椅子重量時發生錯誤: {e}")
        predicted_weight = None
    
    # 10. 保存結果
    save_enhanced_results(optimized_rf_model, y_test, optimized_y_pred, 
                         best_n_estimators, best_max_depth, X.columns, 
                         predicted_weight, args, output_dir)
    
    # 11. 保存詳細分析結果
    save_detailed_analysis_results(optimized_rf_model, y_test, optimized_y_pred, 
                                  best_n_estimators, best_max_depth, X.columns,
                                  learning_curve_data, oob_data, validation_curve_data, 
                                  predicted_weight, args, output_dir)
    
    # 12. 保存原始數據供進一步分析
    save_raw_data_for_analysis(y_test, optimized_y_pred, learning_curve_data, 
                              oob_data, validation_curve_data, predicted_weight, args, output_dir)
    
    # 13. 保存模型
    try:
        import pickle
        with open(os.path.join(output_dir, 'optimized_rf_model.pkl'), 'wb') as f:
            pickle.dump(optimized_rf_model, f)
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(output_dir, 'imputer.pkl'), 'wb') as f:
            pickle.dump(imputer, f)
    except Exception as e:
        print(f"保存模型時發生錯誤: {e}")
    
    print(f"\n增強版隨機森林分析完成！所有結果已保存到: {output_dir}")
    
    return predicted_weight

if __name__ == "__main__":
    main()