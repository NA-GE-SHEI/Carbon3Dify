import pandas as pd
import numpy as np
import os, time
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='椅子重量預測與分析工具')
    
    # 主要參數
    parser.add_argument('--data_file', type=str, default='./models/chair_raw_data_v2_idmatched_revised_cleaned_aug_v2.csv', help='資料檔案路徑')
    parser.add_argument('--encoding', type=str, default='utf-8', help='CSV檔案編碼')
    parser.add_argument('--output_dir', type=str, default='./result', help='輸出目錄')
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
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
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
                print(f"Error drawing chair weight comparison chart: {e}")
                print(f"處理欄位 '{col}' 時發生錯誤: {e}")
    
    # 再次檢查資料型態
    print("\n資料型態檢查：")
    print(df.dtypes)
    
    return df


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


def train_and_evaluate_rf(X_train_scaled, y_train, X_test_scaled, y_test, args):
    """訓練與評估隨機森林模型"""
    print("\n訓練隨機森林模型...")
    
    # 建立隨機森林模型
    rf_model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=args.random_state
    )
    
    # 訓練模型
    rf_model.fit(X_train_scaled, y_train)
    
    # 預測和評估
    y_pred = rf_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"隨機森林模型評估:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    return rf_model, y_pred, rmse, r2


def train_and_evaluate_xgb(X_train_scaled, y_train, X_test_scaled, y_test, args):
    """訓練與評估XGBoost模型"""
    print("\n訓練XGBoost模型...")
    
    # 建立XGBoost模型
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.1,
        random_state=args.random_state
    )
    
    # 訓練模型
    xgb_model.fit(X_train_scaled, y_train)
    
    # 預測和評估
    y_pred = xgb_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"XGBoost模型評估:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    return xgb_model, y_pred, rmse, r2


def train_and_evaluate_mlr(X_train_scaled, y_train, X_test_scaled, y_test, args):
    """訓練與評估多元線性迴歸模型"""
    print("\n訓練多元線性迴歸模型...")
    
    # 建立多元線性迴歸模型
    mlr_model = LinearRegression()
    
    # 訓練模型
    mlr_model.fit(X_train_scaled, y_train)
    
    # 預測和評估
    y_pred = mlr_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"多元線性迴歸模型評估:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    return mlr_model, y_pred, rmse, r2


def train_and_evaluate_svr(X_train_scaled, y_train, X_test_scaled, y_test, args):
    """訓練與評估支持向量迴歸模型"""
    print("\n訓練支持向量迴歸模型...")
    
    # 建立支持向量迴歸模型
    svr_model = SVR(
        kernel=args.svr_kernel,
        C=args.svr_c,
        epsilon=args.svr_epsilon,
        gamma='scale'
    )
    
    # 訓練模型
    svr_model.fit(X_train_scaled, y_train)
    
    # 預測和評估
    y_pred = svr_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"支持向量迴歸模型評估:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    return svr_model, y_pred, rmse, r2


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


def visualize_feature_importance(model, X, output_dir, model_name, args):
    """視覺化特徵重要性"""
    # 對於線性模型，使用係數作為特徵重要性
    if model_name == "多元線性迴歸":
        importance = np.abs(model.coef_)
        feature_names = X.columns
    elif model_name == "支持向量迴歸":
        # SVR沒有內建特徵重要性，我們使用排列重要性或SHAP值的替代方法
        # 這裡簡單實現一個基於係數的方法，僅適用於線性核
        if args.svr_kernel == 'linear':
            # 對於線性核，可以從模型中獲取係數
            importance = np.abs(model.coef_[0])
            feature_names = X.columns
        else:
            print(f"注意: 非線性SVR核函數不支持直接提取特徵重要性。跳過特徵重要性視覺化。")
            return
    else:
        # 對於樹模型，使用原有的特徵重要性
        importance = model.feature_importances_
        feature_names = X.columns
    
    # 按重要性排序
    indices = np.argsort(importance)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_importance)), sorted_importance, align='center', 
            color=cm.Blues(np.linspace(0.4, 0.8, len(sorted_importance))))
    plt.xticks(range(len(sorted_importance)), sorted_feature_names, rotation=45, ha='right')
    plt.title(f'Chair Weight Influencing Factors Analysis ({model_name})', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_feature_importance.png'), dpi=args.dpi)
    plt.close()


def visualize_predictions(y_test, y_pred, output_dir, model_name, args):
    """視覺化實際值與預測值比較"""
    plt.figure(figsize=(10, 8))
    # 修正：使用顏色映射的正確方法
    plt.scatter(y_test, y_pred, alpha=0.7, s=80, color=cm.Blues(0.7))
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Weight (kg)', fontsize=14)
    plt.ylabel('Predicted Weight (kg)', fontsize=14)
    plt.title(f'Actual vs Predicted Weight Comparison ({model_name})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_actual_vs_predicted.png'), dpi=args.dpi)
    plt.close()


def visualize_seat_volume_weight(model, X, scaler, output_dir, model_name, args):
    """視覺化椅墊體積與重量關係"""
    try:
        if X.empty:
            print("錯誤: 資料集為空，無法繪製椅墊體積與重量關係圖")
            return
            
        base_chair = X.iloc[0].copy()
        seat_volumes = np.linspace(500, 3000, 100)
        weights = []
        
        for volume in seat_volumes:
            # 調整椅墊面積以對應不同體積 (維持相同厚度)
            if '椅墊厚度(cm)' not in base_chair:
                print("錯誤: 資料集中沒有'椅墊厚度(cm)'列")
                return
                
            area = volume / base_chair['椅墊厚度(cm)']
            temp_chair = base_chair.copy()
            
            # 修正：確保欄位名稱一致
            if '椅墊面積(cm)' in X.columns:
                temp_chair['椅墊面積(cm)'] = area
            else:
                print(f"警告: '椅墊面積(cm)' 列不存在於資料集中，跳過該列")
                return
            
            # 轉換為DataFrame, 標準化並預測
            temp_df = pd.DataFrame([temp_chair])
            temp_scaled = scaler.transform(temp_df)
            weight = model.predict(temp_scaled)[0]
            weights.append(weight)
        
        plt.figure(figsize=(10, 6))
        plt.plot(seat_volumes, weights, 'b-', lw=2.5)
        
        # 只有在知道真實重量時才繪製參考線
        if args.true_weight is not None:
            plt.axhline(y=args.true_weight, color='g', linestyle='--', label=f'Actual Weight ({args.true_weight} kg)')
            plt.scatter([args.seat_volume/1000], [args.true_weight], s=100, c='red', zorder=5)
        
        # 在任何情況下都顯示當前椅墊體積的位置
        plt.axvline(x=args.seat_volume/1000, color='r', linestyle='--', label=f'Your Chair ({args.seat_volume/1000:.0f} cm³)')
        
        plt.xlabel('Seat Volume (cm³)', fontsize=14)
        plt.ylabel('Chair Weight (kg)', fontsize=14)
        plt.title(f'Impact of Seat Volume on Chair Weight ({model_name})', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_seat_volume_vs_weight.png'), dpi=args.dpi)
        plt.close()
    except Exception as e:
        print(f"Error drawing seat volume vs weight chart: {e}")


def visualize_3d_relationship(model, X, scaler, output_dir, model_name, args):
    """視覺化3D關係 (椅腳體積, 椅墊厚度, 重量)"""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        if X.empty:
            print("錯誤: 資料集為空，無法繪製3D關係圖")
            return
            
        base_chair = X.iloc[0].copy()
        
        # 檢查必要的列是否存在
        if '椅腳體積(cm)' not in X.columns or '椅墊厚度(cm)' not in X.columns:
            print("錯誤: 資料集中缺少必要的列'椅腳體積(cm)'或'椅墊厚度(cm)'")
            return
            
        leg_volumes = np.linspace(1000, 3000, 15)
        seat_thickness = np.linspace(1, 7, 15)
        leg_v, seat_t = np.meshgrid(leg_volumes, seat_thickness)
        weights = np.zeros(leg_v.shape)
        
        for i in range(len(leg_volumes)):
            for j in range(len(seat_thickness)):
                temp_chair = base_chair.copy()
                # 修正：確保欄位名稱一致
                temp_chair['椅腳體積(cm)'] = leg_v[j,i]
                temp_chair['椅墊厚度(cm)'] = seat_t[j,i]
                
                temp_df = pd.DataFrame([temp_chair])
                temp_scaled = scaler.transform(temp_df)
                weights[j,i] = model.predict(temp_scaled)[0]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(leg_v, seat_t, weights, cmap=cm.Blues, 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # 標示當前椅子的位置（如果有真實重量可用）
        predicted_weight = model.predict(scaler.transform(pd.DataFrame([base_chair])))[0]
        
        if args.true_weight is not None:
            ax.scatter([args.leg_volume], [args.seat_thickness], [args.true_weight], 
                      color='red', s=100, label='Your Chair (Actual)')
        else:
            ax.scatter([args.leg_volume], [args.seat_thickness], [predicted_weight], 
                      color='red', s=100, label='Your Chair (Predicted)')
        
        ax.set_xlabel('Leg Volume (cm³)', fontsize=12)
        ax.set_ylabel('Seat Thickness (cm)', fontsize=12)
        ax.set_zlabel('Chair Weight (kg)', fontsize=12)
        ax.set_title(f'Relationship between Leg Volume, Seat Thickness and Chair Weight ({model_name})', fontsize=14)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Weight (kg)')
        plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_3d_weight_analysis.png'), dpi=args.dpi)
        plt.close()
    except Exception as e:
        print(f"Error drawing 3D relationship chart: {e}")


def visualize_chair_comparison(output_dir, args, predicted_weight):
    """視覺化椅子重量比較 - 修改為使用預測重量"""
    try:
        # 椅子列表 - 使用預測的重量
        chairs = [
            {"name": "Your Chair (Predicted)", "weight": predicted_weight}
        ]
        
        # 如果有真實重量，也加入比較
        if args.true_weight is not None:
            chairs.append({"name": "Your Chair (Actual)", "weight": args.true_weight})
        
        # 如果提供了比較椅子資料，則解析並添加
        if args.comparison_chairs:
            chair_data = args.comparison_chairs.split(',')
            for chair in chair_data:
                name, weight = chair.split(':')
                chairs.append({"name": name.strip(), "weight": float(weight.strip())})
        else:
            # 否則使用預設值
            chairs.extend([
                {"name": "Standard Office Chair", "weight": 6.5},
                {"name": "High-back Chair", "weight": 7.8},
                {"name": "Lightweight Dining Chair", "weight": 3.5},
                {"name": "Folding Chair", "weight": 2.8},
                {"name": "Cushioned Armchair", "weight": 8.2}
            ])
        
        plt.figure(figsize=(12, 6))
        names = [chair['name'] for chair in chairs]
        weights = [chair['weight'] for chair in chairs]
        
        # 設定顏色 - 預測值為藍色，真實值(如果有)為綠色
        colors = []
        for name in names:
            if "Predicted" in name:
                colors.append(cm.Blues(0.8))
            elif "Actual" in name:
                colors.append(cm.Greens(0.8))
            else:
                colors.append(cm.Blues(0.5))
        
        bars = plt.bar(names, weights, color=colors)
        
        # 只有在知道真實重量時才繪製參考線
        if args.true_weight is not None:
            plt.axhline(y=args.true_weight, color='g', linestyle='--', alpha=0.7)
        
        plt.xlabel('Chair Type', fontsize=14)
        plt.ylabel('Weight (kg)', fontsize=14)
        plt.title('Chair Weight Comparison', fontsize=16)
        plt.xticks(rotation=15)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'chair_weight_comparison.png'), dpi=args.dpi)
        plt.close()
    except Exception as e:
        print(f"Error drawing chair weight comparison chart: {e}")


def visualize_decision_tree(model, X, output_dir, args):
    """視覺化隨機森林中的單一決策樹"""
    try:
        plt.figure(figsize=(20, 10))
        tree.plot_tree(model.estimators_[0], 
                      feature_names=X.columns,
                      filled=True,
                      rounded=True,
                      max_depth=3)
        plt.title('Single Decision Tree Example from Random Forest', fontsize=16)
        plt.savefig(os.path.join(output_dir, 'decision_tree_example.png'), dpi=args.dpi)
        plt.close()
    except Exception as e:
        print(f"Error drawing decision tree example: {e}")


def visualize_model_comparison(models_results, output_dir, args):
    """視覺化不同模型的評估指標比較"""
    try:
        # 準備數據
        model_names = list(models_results.keys())
        rmse_values = [models_results[name]['rmse'] for name in model_names]
        r2_values = [models_results[name]['r2'] for name in model_names]
        
        # 設置圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # 繪製RMSE對比
        bars1 = ax1.bar(model_names, rmse_values, color=cm.Blues(np.linspace(0.5, 0.9, len(model_names))))
        ax1.set_ylabel('RMSE (Lower is Better)', fontsize=14)
        ax1.set_title('RMSE Comparison of Different Models', fontsize=16)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加RMSE數值標籤
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11)
        
        # 繪製R²對比
        bars2 = ax2.bar(model_names, r2_values, color=cm.Greens(np.linspace(0.5, 0.9, len(model_names))))
        ax2.set_ylabel('R² (Higher is Better)', fontsize=14)
        ax2.set_title('R² Comparison of Different Models', fontsize=16)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加R²數值標籤
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=args.dpi)
        plt.close()
    except Exception as e:
        print(f"Error drawing model comparison chart: {e}")


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
        
    # 訓練模型並預測
    models_results = {}
    
    # 隨機森林
    if args.model_type in ['rf', 'all']:
        rf_model, rf_y_pred, rf_rmse, rf_r2 = train_and_evaluate_rf(X_train_scaled, y_train, X_test_scaled, y_test, args)
        models_results['隨機森林'] = {'model': rf_model, 'y_pred': rf_y_pred, 'rmse': rf_rmse, 'r2': rf_r2}
        
        # 特徵重要性視覺化
        visualize_feature_importance(rf_model, X, output_dir, "隨機森林", args)
        
        # 預測視覺化
        visualize_predictions(y_test, rf_y_pred, output_dir, "隨機森林", args)
        
        try:
            # 椅墊體積與重量關係視覺化
            visualize_seat_volume_weight(rf_model, X, scaler, output_dir, "隨機森林", args)
            
            # 3D關係視覺化
            visualize_3d_relationship(rf_model, X, scaler, output_dir, "隨機森林", args)
        except Exception as e:
            print(f"隨機森林模型視覺化失敗: {e}")
        
        # 決策樹視覺化
        visualize_decision_tree(rf_model, X, output_dir, args)
    
    # XGBoost
    if args.model_type in ['xgb', 'all']:
        xgb_model, xgb_y_pred, xgb_rmse, xgb_r2 = train_and_evaluate_xgb(X_train_scaled, y_train, X_test_scaled, y_test, args)
        models_results['XGBoost'] = {'model': xgb_model, 'y_pred': xgb_y_pred, 'rmse': xgb_rmse, 'r2': xgb_r2}
        
        # 特徵重要性視覺化
        visualize_feature_importance(xgb_model, X, output_dir, "XGBoost", args)
        
        # 預測視覺化
        visualize_predictions(y_test, xgb_y_pred, output_dir, "XGBoost", args)
        
        try:
            # 椅墊體積與重量關係視覺化
            visualize_seat_volume_weight(xgb_model, X, scaler, output_dir, "XGBoost", args)
            
            # 3D關係視覺化
            visualize_3d_relationship(xgb_model, X, scaler, output_dir, "XGBoost", args)
        except Exception as e:
            print(f"XGBoost模型視覺化失敗: {e}")
    
    # 多元線性迴歸
    if args.model_type in ['mlr', 'all']:
        mlr_model, mlr_y_pred, mlr_rmse, mlr_r2 = train_and_evaluate_mlr(X_train_scaled, y_train, X_test_scaled, y_test, args)
        models_results['多元線性迴歸'] = {'model': mlr_model, 'y_pred': mlr_y_pred, 'rmse': mlr_rmse, 'r2': mlr_r2}
        
        # 特徵重要性視覺化
        visualize_feature_importance(mlr_model, X, output_dir, "多元線性迴歸", args)
        
        # 預測視覺化
        visualize_predictions(y_test, mlr_y_pred, output_dir, "多元線性迴歸", args)
        
        try:
            # 椅墊體積與重量關係視覺化
            visualize_seat_volume_weight(mlr_model, X, scaler, output_dir, "多元線性迴歸", args)
            
            # 3D關係視覺化
            visualize_3d_relationship(mlr_model, X, scaler, output_dir, "多元線性迴歸", args)
        except Exception as e:
            print(f"多元線性迴歸模型視覺化失敗: {e}")
    
    # 支持向量迴歸
    if args.model_type in ['svr', 'all']:
        svr_model, svr_y_pred, svr_rmse, svr_r2 = train_and_evaluate_svr(X_train_scaled, y_train, X_test_scaled, y_test, args)
        models_results['支持向量迴歸'] = {'model': svr_model, 'y_pred': svr_y_pred, 'rmse': svr_rmse, 'r2': svr_r2}
        
        # 特徵重要性視覺化 (只有在使用線性核的情況下)
        if args.svr_kernel == 'linear':
            visualize_feature_importance(svr_model, X, output_dir, "支持向量迴歸", args)
        
        # 預測視覺化
        visualize_predictions(y_test, svr_y_pred, output_dir, "支持向量迴歸", args)
        
        try:
            # 椅墊體積與重量關係視覺化
            visualize_seat_volume_weight(svr_model, X, scaler, output_dir, "支持向量迴歸", args)
            
            # 3D關係視覺化
            visualize_3d_relationship(svr_model, X, scaler, output_dir, "支持向量迴歸", args)
        except Exception as e:
            print(f"支持向量迴歸模型視覺化失敗: {e}")
    
    # 模型比較視覺化 (如果有多個模型)
    if len(models_results) > 1:
        visualize_model_comparison(models_results, output_dir, args)
    
    # 新椅子預測
    try:
        new_chair = create_new_chair_data(args)
        
        # 檢查欄位名稱是否一致
        for col in X.columns:
            if col not in new_chair.columns:
                print(f"警告: 新椅子資料中缺少欄位 '{col}'")
                # 添加缺失的欄位
                new_chair[col] = 0
        
        # 確保欄位順序一致
        new_chair = new_chair[X.columns]
        
        # 處理缺失值並標準化
        new_chair_imputed = imputer.transform(new_chair)
        new_chair = pd.DataFrame(new_chair_imputed, columns=X.columns)
        new_chair_scaled = scaler.transform(new_chair)
    except Exception as e:
        print(f"準備新椅子資料時發生錯誤: {e}")
        return None
    
    # 保存參數設定
    with open(os.path.join(output_dir, 'parameters.txt'), 'w', encoding='utf-8') as f:
        f.write("參數設定：\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # 預測結果
    results = {}
    avg_predicted_weight = 0
    
    # 使用所有訓練好的模型進行預測
    for model_name, model_data in models_results.items():
        try:
            model = model_data['model']
            predicted_weight = model.predict(new_chair_scaled)[0]
            
            print(f"\n新椅子的預測重量 ({model_name}): {predicted_weight:.2f} kg")
            results[model_name] = predicted_weight
            avg_predicted_weight += predicted_weight
            
            # 保存模型
            import pickle
            with open(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"使用{model_name}預測新椅子重量時發生錯誤: {e}")
    
    # 計算平均預測重量
    if len(results) > 0:
        avg_predicted_weight /= len(results)
    else:
        print("錯誤: 沒有成功的模型預測結果")
        return None
    
    try:
        # 椅子重量比較視覺化 - 使用平均預測的重量
        visualize_chair_comparison(output_dir, args, avg_predicted_weight)
    except Exception as e:
        print(f"Error drawing chair weight comparison chart: {e}")
    
    # 保存縮放器
    try:
        import pickle
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    except Exception as e:
        print(f"保存縮放器時發生錯誤: {e}")
    
    # 保存結果
    try:
        with open(os.path.join(output_dir, 'results.txt'), 'w', encoding='utf-8') as f:
            f.write("椅子重量預測結果：\n\n")
            
            # 各模型的R²值
            for model_name, model_data in models_results.items():
                f.write(f"{model_name} R²: {model_data['r2']:.6f}\n")
                f.write(f"{model_name} RMSE: {model_data['rmse']:.6f}\n")
            f.write("\n")
            
            # 各模型的預測結果
            for model_name, predicted_weight in results.items():
                f.write(f"{model_name}預測重量: {predicted_weight:.2f} kg\n")
                
                # 如果有真實重量，才計算誤差
                if args.true_weight is not None:
                    f.write(f"誤差: {abs(predicted_weight - args.true_weight):.2f} kg\n")
                    f.write(f"相對誤差: {abs(predicted_weight - args.true_weight) / args.true_weight * 100:.2f}%\n")
                f.write("\n")
            
            # 如果有多個模型，計算平均
            if len(results) > 1:
                f.write(f"平均預測重量: {avg_predicted_weight:.2f} kg\n")
                
                # 如果有真實重量，才計算誤差
                if args.true_weight is not None:
                    f.write(f"平均誤差: {abs(avg_predicted_weight - args.true_weight):.2f} kg\n")
                    f.write(f"平均相對誤差: {abs(avg_predicted_weight - args.true_weight) / args.true_weight * 100:.2f}%\n")
            
            # 如果沒有真實重量，添加說明
            if args.true_weight is None:
                f.write("\n注意：未提供真實重量，因此無法計算預測誤差。\n")
                f.write("如果之後獲得了實際重量，可以手動計算誤差或使用--true_weight參數重新運行分析。\n")
    except Exception as e:
        print(f"保存結果時發生錯誤: {e}")
    
    print(f"\n預測分析完成！所有結果已保存到: {output_dir}")
    
    # 返回平均預測值，方便其他程式呼叫使用
    return avg_predicted_weight if len(results) > 0 else None


if __name__ == "__main__":
    main()
