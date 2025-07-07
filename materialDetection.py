import os
import cv2
import numpy as np
import glob
from mmseg.apis import init_model, inference_model
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import argparse
from collections import Counter
from pathlib import Path

# 定義類別和調色板
classes = ['fabric', 'glass', 'leather', 'metal', 'plastic', 'stone', 'wood']
palette = [
    [80, 50, 50],     # fabric - 棕色
    [140, 140, 140],  # glass - 亮灰色  
    [204, 5, 255],    # leather - 紫色
    [4, 250, 7],      # metal - 绿色
    [8, 255, 51],     # plastic - 亮绿色
    [255, 51, 7],     # stone - 红色
    [255, 255, 0]     # wood - 明亮黃色
]

def load_model(config_path, checkpoint_path):
    """加載訓練好的模型"""
    print(f"🤖 Loading MINC Material Recognition Model...")
    print(f"📁 Config file: {config_path}")
    print(f"⚖️ Weight file: {checkpoint_path}")
    
    try:
        model = init_model(config_path, checkpoint_path, device='cuda:0')
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        print("🔧 Trying CPU mode...")
        try:
            model = init_model(config_path, checkpoint_path, device='cpu')
            print("✅ CPU mode loaded successfully!")
            return model
        except Exception as e2:
            print(f"❌ CPU mode also failed: {str(e2)}")
            return None

def create_labeled_visualization(img_path, pred_mask, save_dir, alpha_mask=None, chair_id=None):
    """創建帶有詳細標籤的可視化結果"""
    # 讀取原圖（保留alpha通道）
    img_rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_rgba is None:
        print(f"❌ Cannot read image: {img_path}")
        return
    
    # 處理不同通道數的圖像
    if img_rgba.shape[2] == 4:  # RGBA
        has_alpha = True
        img_rgb = cv2.cvtColor(img_rgba[:,:,:3], cv2.COLOR_BGR2RGB)
        alpha_channel = img_rgba[:,:,3]
        if alpha_mask is None:
            alpha_mask = alpha_channel > 0  # 非透明區域
    elif img_rgba.shape[2] == 3:  # RGB
        has_alpha = False
        img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2RGB)
        h, w = img_rgba.shape[:2]
        alpha_mask = np.ones((h, w), dtype=bool)  # 全部區域都有效
    
    h, w = pred_mask.shape
    
    # 創建彩色分割圖
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        mask = pred_mask == class_id
        colored_mask[mask] = color
    
    # 如果有透明區域，將其設為灰色
    if has_alpha:
        transparent_mask = ~alpha_mask
        colored_mask[transparent_mask] = [128, 128, 128]  # 灰色表示透明區域
    
    # 統計各類別像素數和百分比（只計算非透明區域）
    if has_alpha:
        # 只統計非透明區域的材料分布
        valid_pred_mask = pred_mask[alpha_mask]
        unique, counts = np.unique(valid_pred_mask, return_counts=True)
        total_valid_pixels = np.sum(alpha_mask)
        total_pixels = pred_mask.size
        transparent_pixels = total_pixels - total_valid_pixels
    else:
        unique, counts = np.unique(pred_mask, return_counts=True)
        total_valid_pixels = pred_mask.size
        total_pixels = pred_mask.size
        transparent_pixels = 0
    
    class_stats = {}
    class_stats_global = {}  # 全圖佔比（包含透明區域）
    
    for class_id, count in zip(unique, counts):
        if class_id < len(classes):
            # 非透明區域中的佔比
            percentage_valid = count / total_valid_pixels * 100 if total_valid_pixels > 0 else 0
            # 全圖佔比（包含透明區域）
            percentage_global = count / total_pixels * 100
            
            class_stats[classes[class_id]] = {
                'pixels': count,
                'percentage': percentage_valid,  # 主要使用非透明區域佔比
                'percentage_global': percentage_global
            }
            class_stats_global[classes[class_id]] = percentage_global
    
    # 創建帶標籤的綜合圖像
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # 改為2x3布局
    
    # 原圖
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 透明區域顯示（如果有的話）
    if has_alpha:
        alpha_visual = np.zeros((h, w, 3), dtype=np.uint8)
        alpha_visual[alpha_mask] = [255, 255, 255]  # 白色=有效區域
        alpha_visual[~alpha_mask] = [255, 0, 0]     # 紅色=透明區域
        axes[0, 1].imshow(alpha_visual)
        axes[0, 1].set_title('Valid Area (White=Valid, Red=Transparent)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
    else:
        axes[0, 1].imshow(img_rgb)
        axes[0, 1].set_title('No Transparency Detected', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
    
    # 純分割結果
    axes[0, 2].imshow(colored_mask)
    axes[0, 2].set_title('Material Segmentation\n(Gray=Transparent)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 疊加結果
    alpha = 0.6
    overlay = cv2.addWeighted(img_rgb, alpha, colored_mask[:,:,:3], 1-alpha, 0)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Overlay Result', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 統計圖表（非透明區域）
    ax_stats = axes[1, 1]
    
    # 繪製統計條形圖（只統計非透明區域）
    if class_stats:
        materials = []
        percentages = []
        colors_norm = []
        
        # 按百分比排序（使用非透明區域佔比）
        sorted_stats = sorted(class_stats.items(), key=lambda x: x[1]['percentage'], reverse=True)
        
        for material, stats in sorted_stats:
            if stats['percentage'] > 0.1:  # 只顯示超過0.1%的類別
                materials.append(material)
                percentages.append(stats['percentage'])
                # 獲取對應顏色
                class_idx = classes.index(material)
                color_norm = [c/255.0 for c in palette[class_idx]]
                colors_norm.append(color_norm)
        
        if materials:
            bars = ax_stats.barh(materials, percentages, color=colors_norm)
            ax_stats.set_xlabel('Percentage in Valid Area (%)', fontsize=12)
            ax_stats.set_title('Material Distribution\n(Non-transparent area only)', fontsize=14, fontweight='bold')
            
            # 在條形圖上添加百分比標籤
            for bar, percentage in zip(bars, percentages):
                width = bar.get_width()
                ax_stats.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                             f'{percentage:.1f}%', ha='left', va='center', fontweight='bold')
        else:
            ax_stats.text(0.5, 0.5, 'No significant\nmaterial detected', 
                         ha='center', va='center', transform=ax_stats.transAxes, fontsize=12)
            ax_stats.set_title('Material Distribution\n(Non-transparent area only)', fontsize=14, fontweight='bold')
    
    ax_stats.grid(True, alpha=0.3)
    
    # 詳細信息文本
    ax_info = axes[1, 2]
    ax_info.axis('off')
    
    info_text = []
    info_text.append("Image Analysis Summary")
    if chair_id:
        info_text.append(f"Chair ID: {chair_id}")
    info_text.append("-" * 30)
    
    if has_alpha:
        transparency_ratio = transparent_pixels / total_pixels * 100
        info_text.append(f"Total pixels: {total_pixels:,}")
        info_text.append(f"Valid pixels: {total_valid_pixels:,}")
        info_text.append(f"Transparent: {transparent_pixels:,} ({transparency_ratio:.1f}%)")
        info_text.append("")
    
    info_text.append("Top Materials (in valid area):")
    
    if class_stats:
        # 顯示前5個材料
        sorted_stats = sorted(class_stats.items(), key=lambda x: x[1]['percentage'], reverse=True)
        for i, (material, stats) in enumerate(sorted_stats[:5]):
            if stats['percentage'] > 0.1:
                info_text.append(f"{material}: {stats['percentage']:.1f}%")
        
        # 特別顯示wood信息
        if 'wood' in class_stats:
            wood_stats = class_stats['wood']
            info_text.append("")
            info_text.append("WOOD Analysis:")
            info_text.append(f"   Pixels: {wood_stats['pixels']:,}")
            info_text.append(f"   In valid area: {wood_stats['percentage']:.1f}%")
            if has_alpha:
                info_text.append(f"   In whole image: {wood_stats['percentage_global']:.1f}%")
    
    # 在圖上顯示信息
    ax_info.text(0.05, 0.95, '\n'.join(info_text), transform=ax_info.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # 添加總體信息
    basename = os.path.splitext(os.path.basename(img_path))[0]
    transparency_info = f" (Transparency: {transparency_ratio:.1f}%)" if has_alpha and transparent_pixels > 0 else ""
    chair_info = f" - {chair_id}" if chair_id else ""
    fig.suptitle(f'Material Analysis: {basename}{chair_info}{transparency_info}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存結果（按椅子ID分類）
    if chair_id:
        chair_save_dir = os.path.join(save_dir, chair_id)
        os.makedirs(chair_save_dir, exist_ok=True)
        save_path = os.path.join(chair_save_dir, f"{basename}_material_analysis.png")
    else:
        save_path = os.path.join(save_dir, f"{basename}_material_analysis.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Detailed analysis chart saved: {save_path}")
    
    return class_stats, total_valid_pixels, transparent_pixels

def process_successful_chair_detection(model, base_dir="./image_identify", save_dir="material_analysis_results"):
    """只處理成功檢測的椅子圖像（identify_pass）"""
    print(f"\n🚀 Starting to process successful chair detection results: {base_dir}")
    
    base_path = Path(base_dir)
    
    # 檢查基礎目錄是否存在
    if not base_path.exists():
        print(f"❌ Base directory does not exist: {base_dir}")
        return
    
    # 查找 identify_pass 資料夾
    pass_dir = base_path / 'identify_pass'
    
    if not pass_dir.exists():
        print(f"❌ identify_pass folder not found")
        return
    
    # 創建結果目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 處理成功檢測結果
    print(f"\n🟢 Processing successful detection results: {pass_dir}")
    results = process_pass_directory(model, pass_dir, save_dir)
    
    # 生成報告
    generate_pass_only_report(results['all_results'], results['wood_detections'], 
                             results['transparency_stats'], results['chair_summary'], save_dir)
    
    print(f"\n🎉 Processing completed!")
    print(f"📁 All results saved to: {save_dir}")
    print(f"📊 Processed {len(results['all_results'])} successfully detected images")

def process_pass_directory(model, directory, save_dir):
    """處理 identify_pass 目錄中的圖像（排除original_img）"""
    directory_path = Path(directory)
    
    # 支援的圖片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG', '*.WEBP']
    
    all_results = []
    wood_detections = []
    transparency_stats = []
    chair_summary = {}
    
    # 遍歷所有子資料夾（排除original_img）
    for chair_folder in directory_path.iterdir():
        if chair_folder.is_dir() and chair_folder.name != 'original_img':
            chair_id = chair_folder.name
            print(f"\n📁 Processing chair folder: {chair_id}")
            
            # 初始化椅子統計
            chair_summary[chair_id] = {'count': 0, 'images': [], 'wood_detected': []}
            
            # 搜索該椅子資料夾中的所有圖片
            image_files = []
            for ext in image_extensions:
                found_files = list(chair_folder.glob(ext))
                image_files.extend(found_files)
            
            if not image_files:
                print(f"  ⚠️ No images found in {chair_folder}")
                continue
            
            print(f"  📸 Found {len(image_files)} successfully detected images")
            chair_summary[chair_id]['count'] = len(image_files)
            
            # 處理該椅子的每張圖片
            for i, img_path in enumerate(image_files, 1):
                print(f"    [{i}/{len(image_files)}] 🔍 Processing: {img_path.name}")
                
                try:
                    # 檢查圖像是否有透明通道
                    img_check = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    has_alpha = img_check.shape[2] == 4 if len(img_check.shape) == 3 else False
                    
                    if has_alpha:
                        alpha_channel = img_check[:,:,3]
                        alpha_mask = alpha_channel > 0
                        total_pixels = alpha_mask.size
                        valid_pixels = np.sum(alpha_mask)
                        transparent_pixels = total_pixels - valid_pixels
                        transparency_ratio = transparent_pixels / total_pixels * 100
                        
                        print(f"      🔲 Transparent background: {transparency_ratio:.1f}%")
                        inference_path = str(img_path)
                    else:
                        alpha_mask = None
                        valid_pixels = img_check.shape[0] * img_check.shape[1]
                        transparent_pixels = 0
                        transparency_ratio = 0
                        inference_path = str(img_path)
                    
                    # 進行推理
                    result = inference_model(model, inference_path)
                    pred_mask = result.pred_sem_seg.data.cpu().numpy()[0]
                    
                    # 處理透明區域
                    if has_alpha and alpha_mask is not None:
                        if alpha_mask.shape != pred_mask.shape:
                            alpha_mask_resized = cv2.resize(alpha_mask.astype(np.uint8), 
                                                           (pred_mask.shape[1], pred_mask.shape[0]), 
                                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                        else:
                            alpha_mask_resized = alpha_mask
                        
                        pred_mask[~alpha_mask_resized] = 0
                    else:
                        alpha_mask_resized = None
                    
                    # 統計各類別
                    if has_alpha and alpha_mask_resized is not None:
                        valid_pred_mask = pred_mask[alpha_mask_resized]
                        unique, counts = np.unique(valid_pred_mask, return_counts=True)
                        total_valid_pixels = np.sum(alpha_mask_resized)
                    else:
                        unique, counts = np.unique(pred_mask, return_counts=True)
                        total_valid_pixels = pred_mask.size
                    
                    pred_stats = dict(zip(unique, counts))
                    
                    # 處理材料統計
                    image_materials = {}
                    wood_percentage_valid = 0
                    
                    for class_id, class_name in enumerate(classes):
                        pixel_count = pred_stats.get(class_id, 0)
                        percentage = pixel_count / total_valid_pixels * 100 if total_valid_pixels > 0 else 0
                        image_materials[class_name] = percentage
                        
                        if class_name == 'wood':
                            wood_percentage_valid = percentage
                        
                        if percentage > 1.0:
                            if class_name == 'wood' and percentage > 5.0:
                                print(f"      🌳 {class_name}: {percentage:.1f}% ⭐")
                                wood_detections.append({
                                    'chair_id': chair_id,
                                    'image': img_path.name,
                                    'wood_percentage_valid': percentage,
                                    'wood_pixels': pixel_count,
                                    'valid_pixels': total_valid_pixels,
                                    'has_transparency': has_alpha,
                                    'transparency_ratio': transparency_ratio
                                })
                                chair_summary[chair_id]['wood_detected'].append({
                                    'image': img_path.name,
                                    'wood_percentage': percentage
                                })
                            else:
                                print(f"      📦 {class_name}: {percentage:.1f}%")
                    
                    # 創建可視化結果
                    class_stats, actual_valid_pixels, actual_transparent_pixels = create_labeled_visualization(
                        str(img_path), pred_mask, save_dir, alpha_mask_resized, chair_id)
                    
                    # 記錄結果
                    result_data = {
                        'chair_id': chair_id,
                        'image': img_path.name,
                        'materials': image_materials,
                        'dominant_material': max(image_materials.items(), key=lambda x: x[1]),
                        'has_transparency': has_alpha,
                        'transparency_ratio': transparency_ratio,
                        'valid_pixels': total_valid_pixels,
                        'wood_percentage_in_valid': wood_percentage_valid
                    }
                    
                    all_results.append(result_data)
                    chair_summary[chair_id]['images'].append(result_data)
                    
                    if has_alpha:
                        transparency_stats.append({
                            'chair_id': chair_id,
                            'image': img_path.name,
                            'transparency_ratio': transparency_ratio,
                            'valid_pixels': total_valid_pixels,
                            'total_pixels': pred_mask.size
                        })
                
                except Exception as e:
                    print(f"      ❌ Processing failed: {str(e)}")
                    continue
    
    return {
        'all_results': all_results,
        'wood_detections': wood_detections,
        'transparency_stats': transparency_stats,
        'chair_summary': chair_summary
    }

def generate_pass_only_report(all_results, wood_detections, transparency_stats, chair_summary, save_dir):
    """生成僅針對成功檢測的分析報告"""
    print(f"\n" + "="*80)
    print(f"📋 Successful Chair Detection Material Recognition Analysis Report")
    print(f"="*80)
    
    if not all_results:
        print("❌ No successfully processed images")
        return
    
    print(f"📊 Processing Statistics:")
    print(f"  ✅ Successfully detected images: {len(all_results)}")
    print(f"  🪑 Number of chairs: {len(chair_summary)}")
    
    # 按椅子ID統計
    print(f"\n🪑 Statistics by Chair ID:")
    for chair_id, stats in sorted(chair_summary.items()):
        image_count = stats['count']
        wood_count = len(stats['wood_detected'])
        
        print(f"  {chair_id}: {image_count} images")
        if wood_count > 0:
            avg_wood = sum(w['wood_percentage'] for w in stats['wood_detected']) / wood_count
            print(f"    🌳 Wood detected: {wood_count} images, average ratio: {avg_wood:.1f}%")
    
    # 材料分布統計
    print(f"\n📊 Material Distribution Statistics:")
    dominant_materials = [r['dominant_material'][0] for r in all_results]
    material_counter = Counter(dominant_materials)
    
    for material, count in material_counter.most_common():
        percentage = count / len(all_results) * 100
        print(f"  {material}: {count} images ({percentage:.1f}%)")
    
    # Wood 特別統計
    if wood_detections:
        print(f"\n🌳 Wood Material Detection Details:")
        print(f"  Images with wood detected: {len(wood_detections)}")
        
        avg_wood_all = sum(w['wood_percentage_valid'] for w in wood_detections) / len(wood_detections)
        print(f"  Average wood ratio: {avg_wood_all:.1f}%")
        
        # 按椅子分組顯示wood檢測
        wood_by_chair = {}
        for detection in wood_detections:
            chair_id = detection['chair_id']
            if chair_id not in wood_by_chair:
                wood_by_chair[chair_id] = []
            wood_by_chair[chair_id].append(detection)
        
        print(f"\n  🪑 Wood detection grouped by chair:")
        for chair_id, detections in sorted(wood_by_chair.items()):
            print(f"    {chair_id}: {len(detections)} images with wood detected")
            avg_wood = sum(d['wood_percentage_valid'] for d in detections) / len(detections)
            print(f"      Average wood ratio: {avg_wood:.1f}%")
            
            # 顯示前3個木材佔比最高的圖像
            top_wood = sorted(detections, key=lambda x: x['wood_percentage_valid'], reverse=True)[:3]
            for detection in top_wood:
                transparency_info = f" (Transparency: {detection['transparency_ratio']:.1f}%)" if detection['has_transparency'] else ""
                print(f"        📸 {detection['image']}: {detection['wood_percentage_valid']:.1f}%{transparency_info}")
    else:
        print(f"\n🌳 Wood Material Detection: No significant wood areas detected (>5%)")
    
    # 透明背景統計
    transparent_images = [r for r in all_results if r.get('has_transparency', False)]
    if transparent_images:
        print(f"\n🔲 Transparent Background Statistics:")
        print(f"  Images with transparent background: {len(transparent_images)}/{len(all_results)} ({len(transparent_images)/len(all_results)*100:.1f}%)")
        
        if transparency_stats:
            avg_transparency = sum(t['transparency_ratio'] for t in transparency_stats) / len(transparency_stats)
            max_transparency = max(t['transparency_ratio'] for t in transparency_stats)
            min_transparency = min(t['transparency_ratio'] for t in transparency_stats)
            print(f"  Average transparency: {avg_transparency:.1f}%")
            print(f"  Transparency range: {min_transparency:.1f}% ~ {max_transparency:.1f}%")
    
    # 保存詳細報告到文件
    save_pass_only_reports(all_results, wood_detections, chair_summary, save_dir)
    
    print(f"\n📄 Detailed reports saved to: {save_dir}")

def save_pass_only_reports(all_results, wood_detections, chair_summary, save_dir):
    """保存僅針對成功檢測的詳細報告"""
    
    # 1. 成功檢測材料分析報告
    report_path = os.path.join(save_dir, "successful_detection_material_analysis.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Successful Chair Detection Material Recognition Analysis Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Processing Statistics:\n")
        f.write(f"- Successfully detected images: {len(all_results)}\n")
        f.write(f"- Number of chairs: {len(chair_summary)}\n\n")
        
        # 按椅子統計
        f.write("Statistics by Chair ID:\n")
        f.write("-"*40 + "\n")
        for chair_id, stats in sorted(chair_summary.items()):
            image_count = stats['count']
            wood_count = len(stats['wood_detected'])
            
            f.write(f"{chair_id}:\n")
            f.write(f"  Image count: {image_count}\n")
            
            if wood_count > 0:
                avg_wood = sum(w['wood_percentage'] for w in stats['wood_detected']) / wood_count
                f.write(f"  Wood detection: {wood_count} images\n")
                f.write(f"  Average wood ratio: {avg_wood:.1f}%\n")
            else:
                f.write(f"  Wood detection: No significant wood detected (>5%)\n")
            f.write("\n")
        
        # 詳細圖像信息
        f.write("Detailed Image Analysis:\n")
        f.write("-"*40 + "\n")
        
        for chair_id in sorted(chair_summary.keys()):
            f.write(f"\n{chair_id}:\n")
            
            chair_results = [r for r in all_results if r['chair_id'] == chair_id]
            
            for result in chair_results:
                f.write(f"  📸 {result['image']}: Primary material={result['dominant_material'][0]} ({result['dominant_material'][1]:.1f}%)")
                if result.get('wood_percentage_in_valid', 0) > 5:
                    f.write(f", Wood={result['wood_percentage_in_valid']:.1f}%")
                if result.get('has_transparency', False):
                    f.write(f", Transparency={result['transparency_ratio']:.1f}%")
                f.write("\n")
    
    # 2. 木材檢測專門報告
    if wood_detections:
        wood_report_path = os.path.join(save_dir, "successful_wood_detection_analysis.txt")
        with open(wood_report_path, 'w', encoding='utf-8') as f:
            f.write("Successful Chair Detection Wood Analysis Report\n")
            f.write("="*40 + "\n\n")
            
            # 按椅子分組
            wood_by_chair = {}
            for detection in wood_detections:
                chair_id = detection['chair_id']
                if chair_id not in wood_by_chair:
                    wood_by_chair[chair_id] = []
                wood_by_chair[chair_id].append(detection)
            
            for chair_id, detections in sorted(wood_by_chair.items()):
                f.write(f"{chair_id} (Wood detected: {len(detections)} images):\n")
                
                for detection in detections:
                    f.write(f"  📸 {detection['image']}: {detection['wood_percentage_valid']:.1f}%")
                    if detection['has_transparency']:
                        f.write(f" (Transparency: {detection['transparency_ratio']:.1f}%)")
                    f.write("\n")
                
                # 統計
                avg_wood = sum(d['wood_percentage_valid'] for d in detections) / len(detections)
                max_wood = max(d['wood_percentage_valid'] for d in detections)
                min_wood = min(d['wood_percentage_valid'] for d in detections)
                f.write(f"  Statistics: Average={avg_wood:.1f}%, Max={max_wood:.1f}%, Min={min_wood:.1f}%\n\n")
    
    # 3. CSV格式的統計報告
    csv_report_path = os.path.join(save_dir, "successful_detection_summary.csv")
    import csv
    with open(csv_report_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['Chair ID', 'Image Name', 'Primary Material', 'Primary Material Ratio', 'Wood Ratio', 
                     'Has Transparent Background', 'Transparency', 'Valid Pixels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in all_results:
            writer.writerow({
                'Chair ID': result['chair_id'],
                'Image Name': result['image'],
                'Primary Material': result['dominant_material'][0],
                'Primary Material Ratio': f"{result['dominant_material'][1]:.1f}%",
                'Wood Ratio': f"{result.get('wood_percentage_in_valid', 0):.1f}%",
                'Has Transparent Background': 'Yes' if result.get('has_transparency', False) else 'No',
                'Transparency': f"{result.get('transparency_ratio', 0):.1f}%",
                'Valid Pixels': result.get('valid_pixels', 0)
            })
    
    # 4. 按椅子分組的材料統計CSV
    chair_summary_csv_path = os.path.join(save_dir, "chair_material_summary.csv")
    with open(chair_summary_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['Chair ID', 'Image Count', 'Wood Detection Count', 'Average Wood Ratio', 'Max Wood Ratio', 
                     'Primary Material Type', 'Transparent Background Images']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for chair_id, stats in sorted(chair_summary.items()):
            chair_results = [r for r in all_results if r['chair_id'] == chair_id]
            wood_detections_chair = [w for w in wood_detections if w['chair_id'] == chair_id]
            transparent_count = len([r for r in chair_results if r.get('has_transparency', False)])
            
            # 找出該椅子最常見的材料
            chair_materials = [r['dominant_material'][0] for r in chair_results]
            most_common_material = Counter(chair_materials).most_common(1)[0][0] if chair_materials else 'N/A'
            
            # 木材統計
            if wood_detections_chair:
                avg_wood = sum(w['wood_percentage_valid'] for w in wood_detections_chair) / len(wood_detections_chair)
                max_wood = max(w['wood_percentage_valid'] for w in wood_detections_chair)
            else:
                avg_wood = 0
                max_wood = 0
            
            writer.writerow({
                'Chair ID': chair_id,
                'Image Count': stats['count'],
                'Wood Detection Count': len(wood_detections_chair),
                'Average Wood Ratio': f"{avg_wood:.1f}%",
                'Max Wood Ratio': f"{max_wood:.1f}%",
                'Primary Material Type': most_common_material,
                'Transparent Background Images': transparent_count
            })
    
    print(f"✅ Successful detection analysis report saved: {report_path}")
    if wood_detections:
        print(f"✅ Wood detection report saved: {wood_report_path}")
    print(f"✅ CSV statistics report saved: {csv_report_path}")
    print(f"✅ Chair material statistics saved: {chair_summary_csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Material Recognition Analysis for Successfully Detected Chairs')
    parser.add_argument('--config', type=str, 
                       default='models/7MINC/7_minc_deeplabv3plusplus.py',
                       help='Model configuration file path')
    parser.add_argument('--checkpoint', type=str, 
                       default='models/7MINC/iter_30000.pth',
                       help='Model weight file path')
    parser.add_argument('--base_dir', type=str, 
                       default='./image_identify',
                       help='Chair detection results base directory path')
    parser.add_argument('--save_dir', type=str, 
                       default='material_analysis',
                       help='Material analysis results save directory')
    
    args = parser.parse_args()
    
    # 檢查文件存在性
    if not os.path.exists(args.config):
        print(f"❌ Configuration file does not exist: {args.config}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Weight file does not exist: {args.checkpoint}")
        return
    
    if not os.path.exists(args.base_dir):
        print(f"❌ Chair detection results directory does not exist: {args.base_dir}")
        print(f"Please run chair detection program first to generate {args.base_dir} directory")
        return
    
    # 檢查 identify_pass 是否存在
    pass_dir = Path(args.base_dir) / 'identify_pass'
    if not pass_dir.exists():
        print(f"❌ Successful detection directory does not exist: {pass_dir}")
        print(f"Please ensure chair detection program has run successfully and generated identify_pass directory")
        return
    
    # 加載模型
    model = load_model(args.config, args.checkpoint)
    if model is None:
        print("❌ Model loading failed, program exit")
        return
    
    print(f"📁 Will process successful chair detection results: {args.base_dir}/identify_pass")
    print(f"📁 Excluding original_img folder")
    print(f"📁 Results will be saved to: {args.save_dir}")
    print(f"🎯 Focus on analyzing material distribution of successfully detected chairs")
    
    # 處理成功檢測的椅子結果
    process_successful_chair_detection(model, args.base_dir, args.save_dir)

if __name__ == "__main__":
    main()
