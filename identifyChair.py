import os
import shutil
import csv
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class ChairDetectionSorter:
    def __init__(self, model_path='yolov10n.pt', confidence_threshold=0.8):
        """
        初始化椅子檢測分類器
        
        Args:
            model_path: YOLOv10模型路徑
            confidence_threshold: 信心度閾值
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.chair_class_id = 56  # COCO數據集中椅子的類別ID
        
        # ROC曲線數據收集
        self.y_true = []  # 真實標籤 (假設所有圖片都應該檢測到椅子)
        self.y_scores = []  # 預測分數 (檢測到椅子的最高信心度)
        self.image_names = []  # 圖片名稱
        
        # 椅子ID統計數據收集
        self.chair_stats = {}  # 用於收集每個椅子ID的詳細統計
        
        # 創建輸出資料夾結構 - 修改為 ./image_identify/ 底下
        self.base_dir = Path('image_identify')
        self.pass_dir = self.base_dir / 'identify_pass'
        self.fail_dir = self.base_dir / 'identify_fail'
        
        # 創建主資料夾和子資料夾
        self.pass_original = self.pass_dir / 'original_img'
        self.fail_original = self.fail_dir / 'original_img'
        
        self.base_dir.mkdir(exist_ok=True)
        self.pass_dir.mkdir(exist_ok=True)
        self.fail_dir.mkdir(exist_ok=True)
        self.pass_original.mkdir(exist_ok=True)
        self.fail_original.mkdir(exist_ok=True)
        
        # 設定英文字體
        self._setup_english_font()
        
    def _setup_english_font(self):
        """
        設定matplotlib的英文字體
        """
        # 常見的英文字體列表
        english_fonts = [
            'Arial',               # Arial
            'Times New Roman',     # Times New Roman
            'Calibri',            # Calibri
            'Helvetica',          # Helvetica
            'DejaVu Sans'         # 備用字體
        ]
        
        # 尋找可用的英文字體
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        selected_font = 'DejaVu Sans'  # 預設字體
        
        for font in english_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        # 設定matplotlib字體
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
        
        print(f"Using font: {selected_font}")
        print("-" * 30)
        
    def detect_and_classify(self, input_dir):
        """
        檢測椅子並根據信心度分類
        
        Args:
            input_dir: 輸入圖片資料夾路徑
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"錯誤：輸入資料夾 {input_dir} 不存在！")
            return
        
        # 支援的圖片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        
        # 統計資料
        total_images = 0
        pass_count = 0
        fail_count = 0
        
        print(f"開始處理資料夾: {input_dir}")
        print(f"信心度閾值: {self.confidence_threshold}")
        print(f"輸出資料夾: {self.base_dir}")
        print("-" * 50)
        
        for chair_folder in sorted(input_path.iterdir()):
            if chair_folder.is_dir():
                chair_id = chair_folder.name
                print(f"處理椅子資料夾: {chair_id}")
                
                # 為每個椅子ID創建對應的輸出資料夾
                chair_pass_dir = self.pass_dir / chair_id
                chair_fail_dir = self.fail_dir / chair_id
                chair_pass_original_dir = self.pass_original / chair_id
                chair_fail_original_dir = self.fail_original / chair_id
                
                # 創建椅子專屬資料夾
                chair_pass_dir.mkdir(exist_ok=True)
                chair_fail_dir.mkdir(exist_ok=True)
                chair_pass_original_dir.mkdir(exist_ok=True)
                chair_fail_original_dir.mkdir(exist_ok=True)
                
                # 初始化該椅子的統計數據
                self.chair_stats[chair_id] = {
                    'total_images': 0,
                    'confidences': [],
                    'success_count': 0,
                    'image_details': []
                }
                
                # 遍歷該椅子資料夾內的所有圖片
                for image_path in chair_folder.iterdir():
                    if image_path.suffix.lower() in image_extensions:
                        total_images += 1
                        self.chair_stats[chair_id]['total_images'] += 1
                        
                        print(f"  處理圖片: {image_path.name}")
                        
                        # 載入原始圖片
                        original_image = cv2.imread(str(image_path))
                        if original_image is None:
                            print(f"    無法讀取圖片，跳過")
                            continue
                        
                        # 進行推理
                        results = self.model(str(image_path), conf=0.1, verbose=False)
                        
                        # 檢查是否檢測到椅子
                        chair_detections = self._get_chair_detections(results[0])
                        
                        # 收集ROC數據 - 假設所有圖片都應該有椅子
                        self.y_true.append(1)  # 真實標籤：這張圖真的有椅子
                        self.image_names.append(image_path.name)
                        
                        # 找出最高信心度的椅子檢測
                        if chair_detections:
                            max_confidence = max([det['confidence'] for det in chair_detections])
                            print(f"    檢測到椅子，最高信心度: {max_confidence:.3f}")
                            
                            # 收集椅子統計數據
                            self.chair_stats[chair_id]['confidences'].append(max_confidence)
                            
                            # 預測分數：檢測到椅子的最高置信度
                            self.y_scores.append(max_confidence)
                            
                            if max_confidence >= self.confidence_threshold:
                                self.chair_stats[chair_id]['success_count'] += 1
                                pass_count += 1
                                print(f"    -> 辨識成功")
                                print(f"       原始圖片複製到: {chair_pass_original_dir}")
                                print(f"       裁剪圖片保存到: {chair_pass_dir}")
                                
                                # 複製原始圖片到pass/original_img/{chair_id}/，裁剪椅子區域保存到pass/{chair_id}/
                                self._copy_image(image_path, chair_pass_original_dir)
                                self._save_cropped_chair(original_image, chair_detections, 
                                                       image_path.name, max_confidence, chair_pass_dir)
                                
                                # 記錄圖片詳細資訊
                                self.chair_stats[chair_id]['image_details'].append({
                                    'image_name': image_path.name,
                                    'confidence': max_confidence,
                                    'result': '成功'
                                })
                            else:
                                fail_count += 1
                                print(f"    -> 辨識失敗")
                                print(f"       原始圖片複製到: {chair_fail_original_dir}")
                                print(f"       帶標籤圖片保存到: {chair_fail_dir}")
                                
                                # 複製原始圖片到fail/original_img/{chair_id}/，保存帶標籤圖片到fail/{chair_id}/
                                self._copy_image(image_path, chair_fail_original_dir)
                                self._save_labeled_image(original_image, results[0], image_path.name, chair_fail_dir)
                                
                                # 記錄圖片詳細資訊
                                self.chair_stats[chair_id]['image_details'].append({
                                    'image_name': image_path.name,
                                    'confidence': max_confidence,
                                    'result': '失敗'
                                })
                        else:
                            # 未檢測到椅子
                            print(f"    未檢測到椅子")
                            # 預測分數：未檢測到椅子，分數為0
                            self.y_scores.append(0.0)
                            self.chair_stats[chair_id]['confidences'].append(0.0)
                            
                            print(f"    -> 未檢測到椅子")
                            print(f"       原始圖片複製到: {chair_fail_original_dir}")
                            print(f"       帶標籤圖片保存到: {chair_fail_dir}")
                            
                            self._copy_image(image_path, chair_fail_original_dir)
                            self._save_labeled_image(original_image, results[0], image_path.name, chair_fail_dir)
                            fail_count += 1
                            
                            # 記錄圖片詳細資訊
                            self.chair_stats[chair_id]['image_details'].append({
                                'image_name': image_path.name,
                                'confidence': 0.0,
                                'result': '未檢測'
                            })
                        
                        print()
        
        # 輸出統計結果並進行分析
        self._print_summary(total_images, pass_count, fail_count)
        
        # 生成椅子分析報告 - 保存到image_identify資料夾中
        self._generate_chair_analysis_report()
        
        # 進行置信度分析（更適合當前使用情境）
        self._analyze_confidence_distribution()
        
        # 繪製精準度-召回率曲線（更適合椅子檢測任務）
        self._plot_precision_recall_curve()
        
        # 如果用戶有負樣本，可以繪製ROC曲線
        if self._has_negative_samples():
            self._plot_roc_curve()
        else:
            print("\n" + "💡 " + "="*50)
            print("關於ROC曲線的說明：")
            print("由於所有圖片都是椅子圖片（正樣本），無法繪製傳統ROC曲線")
            print("❌ 不建議將低置信度結果當作負樣本，因為：")
            print("   1. 會造成循環邏輯（用閾值定義標籤，再評估閾值）")
            print("   2. 低置信度的椅子圖片仍然是椅子")
            print("   3. 會產生誤導性的評估結果")
            print("✅ 建議使用上方的精準度-召回率曲線和置信度分析")
            print("="*60)
    
    def _copy_image(self, source_path, destination_dir):
        """
        複製原始圖片到目標資料夾
        
        Args:
            source_path: 源圖片路徑
            destination_dir: 目標資料夾
        """
        destination_path = destination_dir / source_path.name
        shutil.copy2(str(source_path), str(destination_path))
    
    def _get_chair_detections(self, result):
        """
        獲取椅子的檢測結果
        
        Args:
            result: YOLO檢測結果
            
        Returns:
            list: 椅子檢測的詳細資訊列表
        """
        chair_detections = []
        
        if result.boxes is not None:
            boxes = result.boxes
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            for i, class_id in enumerate(classes):
                if int(class_id) == self.chair_class_id:
                    chair_detections.append({
                        'confidence': float(confidences[i]),
                        'bbox': xyxy[i],  # [x1, y1, x2, y2]
                        'class_id': int(class_id)
                    })
        
        return chair_detections
    
    def _save_cropped_chair(self, image, chair_detections, filename, max_confidence, output_dir):
        """
        保存裁剪的椅子區域
        
        Args:
            image: 原始圖片
            chair_detections: 椅子檢測結果
            filename: 檔案名稱
            max_confidence: 最高信心度
            output_dir: 輸出資料夾路徑
        """
        # 找到信心度最高的椅子
        best_chair = max(chair_detections, key=lambda x: x['confidence'])
        bbox = best_chair['bbox'].astype(int)
        
        # 裁剪椅子區域
        x1, y1, x2, y2 = bbox
        # 確保座標在圖片範圍內
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        cropped_chair = image[y1:y2, x1:x2]
        
        # 生成新檔名（包含精度資訊）
        name_parts = Path(filename).stem, f"conf_{max_confidence:.3f}", Path(filename).suffix
        new_filename = f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
        
        # 保存裁剪的椅子圖片到指定的椅子資料夾
        save_path = output_dir / new_filename
        cv2.imwrite(str(save_path), cropped_chair)
        
        print(f"      裁剪區域: ({x1},{y1}) -> ({x2},{y2})")
        print(f"      保存為: {new_filename}")
    
    def _save_labeled_image(self, image, result, filename, output_dir):
        """
        保存帶標籤的圖片
        
        Args:
            image: 原始圖片
            result: YOLO檢測結果
            filename: 檔案名稱
            output_dir: 輸出資料夾路徑
        """
        labeled_image = image.copy()
        
        # 繪製所有檢測框
        if result.boxes is not None:
            boxes = result.boxes
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            for i, (class_id, conf, bbox) in enumerate(zip(classes, confidences, xyxy)):
                x1, y1, x2, y2 = bbox.astype(int)
                class_name = self.model.names[int(class_id)]
                
                # 選擇顏色（椅子用紅色，其他用綠色）
                color = (0, 0, 255) if int(class_id) == self.chair_class_id else (0, 255, 0)
                
                # 繪製邊界框
                cv2.rectangle(labeled_image, (x1, y1), (x2, y2), color, 2)
                
                # 繪製標籤
                label = f"{class_name}: {conf:.3f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(labeled_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(labeled_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 保存帶標籤的圖片到指定的椅子資料夾
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), labeled_image)
        
        print(f"      保存帶標籤圖片: {filename}")
    
    def _print_summary(self, total, pass_count, fail_count):
        """
        輸出統計摘要
        """
        print("=" * 60)
        print("處理完成！統計結果:")
        print(f"總圖片數量: {total}")
        print(f"辨識成功 (>={self.confidence_threshold}): {pass_count}")
        print(f"辨識失敗 (<{self.confidence_threshold} 或未檢測到椅子): {fail_count}")
        
        if total > 0:
            success_rate = pass_count / total * 100
            print(f"辨識成功率: {success_rate:.1f}%")
        
        print("=" * 60)
        print(f"輸出結構:")
        print(f" 📁 {self.base_dir}/")
        print(f"   ├── identify_pass/")
        print(f"   │   ├── original_img/")
        print(f"   │   │   ├── Chair1/ (原始成功圖片)")
        print(f"   │   │   ├── Chair2/ (原始成功圖片)")
        print(f"   │   │   └── ...")
        print(f"   │   ├── Chair1/ (裁剪成功圖片)")
        print(f"   │   ├── Chair2/ (裁剪成功圖片)")
        print(f"   │   └── ...")
        print(f"   └── identify_fail/")
        print(f"       ├── original_img/")
        print(f"       │   ├── Chair1/ (原始失敗圖片)")
        print(f"       │   ├── Chair2/ (原始失敗圖片)")
        print(f"       │   └── ...")
        print(f"       ├── Chair1/ (帶標籤失敗圖片)")
        print(f"       ├── Chair2/ (帶標籤失敗圖片)")
        print(f"       └── ...")
        print("注意：原始圖片保留在原位置，處理結果複製到對應資料夾")
    
    def _plot_roc_curve(self):
        """
        繪製ROC曲線並保存到image_identify資料夾
        """
        if len(self.y_true) == 0:
            print("No detection data available for ROC curve plotting")
            return
        
        # 檢查是否有足夠的數據點
        unique_labels = set(self.y_true)
        if len(unique_labels) < 2:
            print(f"Insufficient data for ROC curve, only single label: {unique_labels}")
            print("Need both successful and failed detection samples")
            return
        
        print("\n" + "=" * 60)
        print("Plotting ROC curve...")
        
        # 轉換為numpy數組
        y_true = np.array(self.y_true)
        y_scores = np.array(self.y_scores)
        
        # 統計數據
        success_count = np.sum(y_true == 1)
        fail_count = np.sum(y_true == 0)
        print(f"Successful detection samples: {success_count}")
        print(f"Failed detection samples: {fail_count}")
        
        # 計算ROC曲線
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # 創建圖表
        plt.figure(figsize=(12, 8))
        
        # 繪製ROC曲線
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        
        # 繪製對角線（隨機分類器）
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier (AUC = 0.5)')
        
        # 標記當前閾值點
        current_threshold_idx = np.argmin(np.abs(thresholds - self.confidence_threshold))
        if current_threshold_idx < len(fpr):
            plt.plot(fpr[current_threshold_idx], tpr[current_threshold_idx], 'bs', markersize=8,
                    label=f'Current Threshold = {self.confidence_threshold}')
        
        # 標記最佳閾值點（最大化Youden指數）
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
        
        # 設定圖表屬性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Chair Detection ROC Curve Analysis', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 添加統計資訊
        stats_text = f'Total Samples: {len(y_true)}\n'
        stats_text += f'Successful Detection: {success_count}\n'
        stats_text += f'Failed Detection: {fail_count}\n'
        stats_text += f'Current Threshold: {self.confidence_threshold}\n'
        stats_text += f'AUC Score: {roc_auc:.3f}\n'
        stats_text += f'Suggested Threshold: {optimal_threshold:.3f}'
        
        plt.text(0.6, 0.15, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # 保存圖表到image_identify資料夾
        save_path = self.base_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ ROC curve saved to: {save_path}")
        print(f"📊 AUC Score: {roc_auc:.3f}")
        print(f"🎯 Suggested optimal threshold: {optimal_threshold:.3f}")
        print(f"⚙️  Current threshold: {self.confidence_threshold}")
        
        # 輸出閾值分析
        self._analyze_thresholds(fpr, tpr, thresholds)
        
        return True
    
    def _generate_chair_analysis_report(self):
        """
        生成詳細的椅子分析報告並保存到image_identify資料夾
        """
        if not self.chair_stats:
            print("沒有椅子統計數據可生成報告")
            return
            
        print("\n" + "📋 生成椅子分析報告...")
        print("="*80)
        
        # 準備報告數據
        report_data = []
        
        for chair_id, stats in sorted(self.chair_stats.items()):
            if stats['total_images'] > 0:
                confidences = stats['confidences']
                avg_confidence = np.mean(confidences) if confidences else 0.0
                max_confidence = np.max(confidences) if confidences else 0.0
                success_rate = stats['success_count'] / stats['total_images'] * 100
                
                report_data.append({
                    'chair_id': chair_id,
                    'total_images': stats['total_images'],
                    'avg_confidence': avg_confidence,
                    'max_confidence': max_confidence,
                    'success_rate': success_rate,
                    'success_count': stats['success_count']
                })
        
        # 生成控制台報告
        print(f"{'椅子ID':<10} {'照片數量':<8} {'平均置信度':<12} {'最高置信度':<12} {'檢測成功率':<12} {'成功數量':<8}")
        print("-"*80)
        
        total_images_all = 0
        total_success_all = 0
        all_confidences = []
        
        for data in report_data:
            print(f"{data['chair_id']:<10} {data['total_images']:<8} "
                  f"{data['avg_confidence']:<12.3f} {data['max_confidence']:<12.3f} "
                  f"{data['success_rate']:<12.1f}% {data['success_count']:<8}")
            
            total_images_all += data['total_images']
            total_success_all += data['success_count']
            
            # 收集所有置信度
            chair_confidences = self.chair_stats[data['chair_id']]['confidences']
            all_confidences.extend(chair_confidences)
        
        print("-"*80)
        overall_success_rate = total_success_all / total_images_all * 100 if total_images_all > 0 else 0
        overall_avg_confidence = np.mean(all_confidences) if all_confidences else 0
        overall_max_confidence = np.max(all_confidences) if all_confidences else 0
        
        print(f"{'總計':<10} {total_images_all:<8} "
              f"{overall_avg_confidence:<12.3f} {overall_max_confidence:<12.3f} "
              f"{overall_success_rate:<12.1f}% {total_success_all:<8}")
        
        # 生成CSV報告
        self._save_csv_report(report_data)
        
        # 生成詳細文字報告
        self._save_detailed_report(report_data)
        
        # 找出表現最好和最差的椅子
        self._analyze_chair_performance(report_data)
        
    def _save_csv_report(self, report_data):
        """
        保存CSV格式的椅子分析報告到image_identify資料夾
        """
        csv_filename = self.base_dir / 'chair_analysis_report.csv'
        
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['Chair ID', 'Image Count', 'Avg Confidence', 'Max Confidence', 'Success Rate(%)', 'Success Count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for data in report_data:
                writer.writerow({
                    'Chair ID': data['chair_id'],
                    'Image Count': data['total_images'],
                    'Avg Confidence': f"{data['avg_confidence']:.3f}",
                    'Max Confidence': f"{data['max_confidence']:.3f}",
                    'Success Rate(%)': f"{data['success_rate']:.1f}",
                    'Success Count': data['success_count']
                })
        
        print(f"✅ CSV report saved to: {csv_filename}")
        
    def _save_detailed_report(self, report_data):
        """
        保存詳細的文字報告到image_identify資料夾
        """
        report_filename = self.base_dir / 'chair_detailed_analysis.txt'
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("Chair Detection Detailed Analysis Report\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Detection Parameters:\n")
            f.write(f"- Model: YOLOv10\n")
            f.write(f"- Confidence Threshold: {self.confidence_threshold}\n")
            f.write(f"- Generated Time: {self._get_current_time()}\n\n")
            
            # 整體統計
            total_images = sum(data['total_images'] for data in report_data)
            total_success = sum(data['success_count'] for data in report_data)
            overall_success_rate = total_success / total_images * 100 if total_images > 0 else 0
            
            f.write(f"Overall Statistics:\n")
            f.write(f"- Total Chair Count: {len(report_data)}\n")
            f.write(f"- Total Image Count: {total_images}\n")
            f.write(f"- Successful Detection Count: {total_success}\n")
            f.write(f"- Overall Success Rate: {overall_success_rate:.1f}%\n\n")
            
            # 各椅子詳細資訊
            f.write("Detailed Analysis by Chair:\n")
            f.write("-"*60 + "\n")
            
            for data in report_data:
                chair_id = data['chair_id']
                f.write(f"\n{chair_id}:\n")
                f.write(f"  Image Count: {data['total_images']}\n")
                f.write(f"  Average Confidence: {data['avg_confidence']:.3f}\n")
                f.write(f"  Maximum Confidence: {data['max_confidence']:.3f}\n")
                f.write(f"  Detection Success Rate: {data['success_rate']:.1f}%\n")
                f.write(f"  Success/Total: {data['success_count']}/{data['total_images']}\n")
                
                # 顯示每張圖片的詳細資訊
                if chair_id in self.chair_stats:
                    f.write(f"  Image Details:\n")
                    for img_detail in self.chair_stats[chair_id]['image_details']:
                        f.write(f"    - {img_detail['image_name']}: "
                               f"Confidence={img_detail['confidence']:.3f}, "
                               f"Result={img_detail['result']}\n")
        
        print(f"✅ Detailed report saved to: {report_filename}")
        
    def _analyze_chair_performance(self, report_data):
        """
        分析椅子檢測效果
        """
        if not report_data:
            return
            
        print(f"\n🎯 椅子檢測效果分析:")
        print("-"*40)
        
        # 按成功率排序
        sorted_by_success = sorted(report_data, key=lambda x: x['success_rate'], reverse=True)
        
        # 最佳表現
        best_chair = sorted_by_success[0]
        print(f"🥇 檢測成功率最高: {best_chair['chair_id']} ({best_chair['success_rate']:.1f}%)")
        
        # 最差表現
        worst_chair = sorted_by_success[-1]
        print(f"🔴 檢測成功率最低: {worst_chair['chair_id']} ({worst_chair['success_rate']:.1f}%)")
        
        # 按平均置信度排序
        sorted_by_confidence = sorted(report_data, key=lambda x: x['avg_confidence'], reverse=True)
        highest_conf_chair = sorted_by_confidence[0]
        print(f"💪 平均置信度最高: {highest_conf_chair['chair_id']} ({highest_conf_chair['avg_confidence']:.3f})")
        
        # 需要關注的椅子（成功率低於80%）
        low_performance_chairs = [data for data in report_data if data['success_rate'] < 80]
        if low_performance_chairs:
            print(f"\n⚠️  需要關注的椅子 (成功率 < 80%):")
            for chair in low_performance_chairs:
                print(f"   {chair['chair_id']}: {chair['success_rate']:.1f}%")
        else:
            print(f"\n✅ 所有椅子檢測成功率都 ≥ 80%")
            
    def _get_current_time(self):
        """
        獲取當前時間字符串
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _has_negative_samples(self):
        """
        檢查是否有負樣本（真實標籤為0的樣本）
        """
        return len(set(self.y_true)) > 1 and 0 in self.y_true
    
    def _analyze_confidence_distribution(self):
        """
        分析置信度分佈，幫助優化閾值選擇
        """
        if len(self.y_scores) == 0:
            print("No detection data for analysis")
            return
            
        print("\n" + "📊 Confidence Distribution Analysis")
        print("="*60)
        
        y_scores = np.array(self.y_scores)
        
        # 基本統計
        print(f"📈 Confidence Statistics:")
        print(f"   Mean: {np.mean(y_scores):.3f}")
        print(f"   Median: {np.median(y_scores):.3f}")
        print(f"   Standard Deviation: {np.std(y_scores):.3f}")
        print(f"   Minimum: {np.min(y_scores):.3f}")
        print(f"   Maximum: {np.max(y_scores):.3f}")
        
        # 不同閾值下的統計
        print(f"\n🎯 Detection Results at Different Thresholds:")
        print(f"{'Threshold':<10} {'Success Count':<14} {'Success Rate':<14} {'Recommendation':<20}")
        print("-"*65)
        
        thresholds_to_test = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_threshold = None
        best_detection_rate = 0
        
        for threshold in thresholds_to_test:
            detected_count = np.sum(y_scores >= threshold)
            detection_rate = detected_count / len(y_scores)
            
            recommendation = ""
            if detection_rate >= 0.9:
                recommendation = "Recommended ✅"
                if detection_rate > best_detection_rate:
                    best_detection_rate = detection_rate
                    best_threshold = threshold
            elif detection_rate >= 0.7:
                recommendation = "Consider ⚠️"
            else:
                recommendation = "Too Strict ❌"
                
            print(f"{threshold:<10.1f} {detected_count:<14} {detection_rate:<14.1%} {recommendation:<20}")
        
        print("-"*65)
        if best_threshold:
            print(f"💡 Suggested Threshold: {best_threshold} (Success Rate: {best_detection_rate:.1%})")
        
        # 繪製置信度分佈圖
        self._plot_confidence_distribution(y_scores)
    
    def _plot_confidence_distribution(self, y_scores):
        """
        繪製置信度分佈直方圖並保存到image_identify資料夾
        """
        plt.figure(figsize=(12, 6))
        
        # 子圖1：直方圖
        plt.subplot(1, 2, 1)
        plt.hist(y_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=self.confidence_threshold, color='red', linestyle='--', 
                   label=f'Current Threshold = {self.confidence_threshold}')
        plt.xlabel('Detection Confidence', fontsize=12)
        plt.ylabel('Image Count', fontsize=12)
        plt.title('Confidence Distribution Histogram', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子圖2：累積分佈
        plt.subplot(1, 2, 2)
        sorted_scores = np.sort(y_scores)
        cumulative_prob = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        plt.plot(sorted_scores, cumulative_prob, 'b-', linewidth=2)
        plt.axvline(x=self.confidence_threshold, color='red', linestyle='--', 
                   label=f'Current Threshold = {self.confidence_threshold}')
        plt.xlabel('Detection Confidence', fontsize=12)
        plt.ylabel('Cumulative Probability', fontsize=12)
        plt.title('Confidence Cumulative Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表到image_identify資料夾
        save_path = self.base_dir / 'confidence_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Confidence distribution plot saved to: {save_path}")
        
        # 分析置信度過低的圖片
        low_confidence_threshold = 0.3
        low_confidence_images = [
            (name, score) for name, score in zip(self.image_names, y_scores) 
            if score < low_confidence_threshold
        ]
        
        if low_confidence_images:
            print(f"\n⚠️  Images with confidence below {low_confidence_threshold}:")
            for name, score in sorted(low_confidence_images, key=lambda x: x[1]):
                print(f"   {name}: {score:.3f}")
        else:
            print(f"\n✅ All images have confidence above {low_confidence_threshold}")
    
    def _plot_precision_recall_curve(self):
        """
        繪製精準度-召回率曲線（適合椅子檢測任務）並保存到image_identify資料夾
        """
        if len(self.y_scores) == 0:
            print("No detection data for analysis")
            return
            
        print("\n" + "📈 Precision-Recall Analysis")
        print("="*60)
        
        y_scores = np.array(self.y_scores)
        
        # 對於椅子檢測任務，所有圖片都是椅子（正樣本）
        # 我們評估不同閾值下的檢測召回率
        thresholds = np.linspace(0.1, 1.0, 100)
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            # 在此閾值下檢測為椅子的圖片數
            detected_as_chair = np.sum(y_scores >= threshold)
            total_chairs = len(y_scores)  # 所有圖片都是椅子
            
            # 計算精準度和召回率
            if detected_as_chair > 0:
                precision = 1.0  # 檢測為椅子的都確實是椅子
                recall = detected_as_chair / total_chairs  # 成功檢測的椅子比例
            else:
                precision = 0.0
                recall = 0.0
                
            precisions.append(precision)
            recalls.append(recall)
        
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        # 計算平均精準度
        ap_score = np.trapz(precisions, recalls)
        
        # 繪製精準度-召回率曲線
        plt.figure(figsize=(10, 8))
        plt.plot(recalls, precisions, color='blue', lw=2, 
                label=f'Precision-Recall Curve (AP = {ap_score:.3f})')
        
        # 標記當前閾值點
        current_detected = np.sum(y_scores >= self.confidence_threshold)
        current_recall = current_detected / len(y_scores)
        current_precision = 1.0 if current_detected > 0 else 0.0
        
        plt.plot(current_recall, current_precision, 'rs', markersize=10,
                label=f'Current Threshold = {self.confidence_threshold} (Recall={current_recall:.2f})')
        
        # 標記建議的閾值點（召回率90%對應的閾值）
        target_recall = 0.9
        target_detected = int(target_recall * len(y_scores))
        if target_detected > 0:
            sorted_scores = np.sort(y_scores)[::-1]  # 降序排列
            if target_detected <= len(sorted_scores):
                suggested_threshold = sorted_scores[target_detected-1]
                plt.plot(target_recall, 1.0, 'go', markersize=8,
                        label=f'Suggested Threshold = {suggested_threshold:.3f} (90% Recall)')
        
        # 設定圖表屬性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall - Proportion of Successfully Detected Chairs', fontsize=12)
        plt.ylabel('Precision - Accuracy of Detection Results', fontsize=12)
        plt.title('Chair Detection Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 添加說明文字
        explanation_text = "Curve Interpretation:\n"
        explanation_text += "• Horizontal line indicates all detections are correct\n"
        explanation_text += "• Higher recall means more chairs detected\n"
        explanation_text += "• Balance recall and detection success rate when choosing threshold"
        
        plt.text(0.02, 0.3, explanation_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        # 保存圖表到image_identify資料夾
        save_path = self.base_dir / 'precision_recall_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Precision-Recall curve saved to: {save_path}")
        print(f"📊 Average Precision Score: {ap_score:.3f}")
        print(f"🎯 Current threshold {self.confidence_threshold} recall: {current_recall:.1%}")
        
        # 提供閾值建議
        print(f"\n💡 Threshold Recommendations:")
        recall_targets = [0.95, 0.90, 0.85, 0.80]
        for target in recall_targets:
            target_detected = int(target * len(y_scores))
            if target_detected > 0 and target_detected <= len(y_scores):
                sorted_scores = np.sort(y_scores)[::-1]
                threshold_needed = sorted_scores[target_detected-1]
                print(f"   {target:.0%} Recall: Threshold ≥ {threshold_needed:.3f}")
        
        return True  # 成功產出ROC曲線
        
    def _analyze_thresholds(self, fpr, tpr, thresholds):
        """
        分析不同閾值的效果
        
        Args:
            fpr: 偽陽性率
            tpr: 真陽性率  
            thresholds: 閾值
        """
        print("\nThreshold Analysis:")
        print("-" * 50)
        print(f"{'Threshold':<10} {'True Positive Rate':<18} {'False Positive Rate':<19} {'Accuracy Est.':<15}")
        print("-" * 50)
        
        # 選擇幾個關鍵閾值進行分析
        key_thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
        
        for threshold in key_thresholds:
            # 找到最接近的閾值索引
            idx = np.argmin(np.abs(thresholds - threshold))
            accuracy_est = (tpr[idx] + (1 - fpr[idx])) / 2  # 簡化的準確度估計
            
            print(f"{threshold:<10.1f} {tpr[idx]:<18.3f} {fpr[idx]:<19.3f} {accuracy_est:<15.3f}")
        
        print("-" * 50)
    
    def verify_roc_output(self):
        """
        驗證ROC曲線是否會正確產出
        """
        print("\n" + "🔍 ROC Curve Output Verification:")
        print("-" * 40)
        
        if len(self.y_true) == 0:
            print("❌ No detection data collected")
            return False
            
        unique_labels = set(self.y_true)
        print(f"📊 Collected label types: {unique_labels}")
        
        if len(unique_labels) < 2:
            print("❌ Insufficient label types for ROC curve")
            print("   Need both successful(1) and failed(0) detection samples")
            return False
        
        success_count = sum(1 for x in self.y_true if x == 1)
        fail_count = sum(1 for x in self.y_true if x == 0)
        
        print(f"✅ Successful detection samples: {success_count}")
        print(f"✅ Failed detection samples: {fail_count}")
        print(f"✅ Total samples: {len(self.y_true)}")
        print("✅ ROC curve output conditions met")
        
        # 檢查文件是否會被產出
        roc_file = self.base_dir / 'roc_curve.png'
        print(f"📁 ROC file will be saved as: {roc_file.absolute()}")
        
        return True

def main():
    # 設定參數
    input_directory = "original_data"  # 請修改為您的輸入圖片資料夾路徑
    
    # 檢查輸入資料夾是否存在
    if not os.path.exists(input_directory):
        print(f"請先創建輸入資料夾: {input_directory}")
        return
    
    # 創建檢測器
    detector = ChairDetectionSorter(
        model_path= r'./models/yolov10m.pt',
        confidence_threshold=0.7
    )
    
    # 執行檢測和分類
    detector.detect_and_classify(input_directory)

if __name__ == "__main__":
    main()
