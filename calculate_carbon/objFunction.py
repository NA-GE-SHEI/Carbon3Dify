import numpy as np
import math
import sys
import os

def read_obj(file_path):
    """
    讀取OBJ文件並返回頂點坐標
    """
    vertices = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('v '):  # 頂點數據
                    parts = line.strip().split()
                    if len(parts) >= 4:  # 確保有x, y, z座標
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except Exception as e:
        print(f"讀取OBJ文件時出錯: {e}")
        return None
    
    return np.array(vertices)

def calculate_model_dimensions(vertices):
    """
    計算模型的尺寸
    """
    if len(vertices) == 0:
        return None
    
    # 計算每個軸的最小和最大值
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    
    # 計算每個軸的尺寸
    dimensions = max_vals - min_vals
    
    # 計算對角線長度 (3D空間中兩點的距離)
    diagonal_length = np.linalg.norm(dimensions)
    
    result = {
        'min_coords': min_vals,
        'max_coords': max_vals,
        'width': dimensions[0],  # X軸
        'height': dimensions[1],  # Y軸
        'depth': dimensions[2],   # Z軸
        'diagonal': diagonal_length,
        'longest_axis': dimensions.max(),
        'longest_axis_name': ['X', 'Y', 'Z'][np.argmax(dimensions)]
    }
    
    return result

def analyze_obj_model(file_path):
    """
    分析OBJ模型並打印尺寸信息
    """
    if not os.path.exists(file_path):
        print(f"錯誤: 找不到文件 '{file_path}'")
        return
    
    print(f"分析OBJ模型: {file_path}")
    
    # 讀取模型頂點
    vertices = read_obj(file_path)
    
    if vertices is None or len(vertices) == 0:
        print("錯誤: 無法讀取頂點數據或模型不包含頂點。")
        return
    
    print(f"模型包含 {len(vertices)} 個頂點")
    
    # 計算尺寸
    dimensions = calculate_model_dimensions(vertices)
    
    if dimensions:
        print("\n模型尺寸信息:")
        print(f"X軸最小值/最大值: {dimensions['min_coords'][0]:.4f} / {dimensions['max_coords'][0]:.4f}")
        print(f"Y軸最小值/最大值: {dimensions['min_coords'][1]:.4f} / {dimensions['max_coords'][1]:.4f}")
        print(f"Z軸最小值/最大值: {dimensions['min_coords'][2]:.4f} / {dimensions['max_coords'][2]:.4f}")
        print("\n尺寸摘要:")
        print(f"寬度 (X軸): {dimensions['width']:.4f}")
        print(f"高度 (Y軸): {dimensions['height']:.4f}")
        print(f"深度 (Z軸): {dimensions['depth']:.4f}")
        print(f"最長軸: {dimensions['longest_axis_name']} 軸，長度: {dimensions['longest_axis']:.4f}")
        print(f"對角線長度: {dimensions['diagonal']:.4f} (模型的最大尺寸)")
    else:
        print("無法計算模型尺寸。")

if __name__ == "__main__":
    analyze_obj_model("Chair_legs.obj")
    # if len(sys.argv) != 2:
    #     print("使用方式: python model_size.py 模型文件.obj")
    # else:
    #     file_path = sys.argv[1]
    #     analyze_obj_model(file_path)