import os
import sys
import subprocess
import logging
import time
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import olca

load_dotenv(r"./.env", override=True)

yolo = os.path.expanduser(os.getenv("YOLO_PYTHON"))
mmsegmentation = os.path.expanduser(os.getenv("MM_PYTHON"))
trellis = os.path.expanduser(os.getenv("TRELLIS_PYTHON"))

# 添加必要的路徑
sys.path.append(str(Path(__file__).parent))

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chair_workflow.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ChairAnalysisWorkflow:
    """椅子分析完整工作流程管理器"""
    
    def __init__(self, base_output_dir: str = "./"):
        """初始化工作流程"""
        self.base_output_dir = Path(base_output_dir)
        self.start_time = time.time()
        self.results = {}
        self.timing_records = {}
        
        # 定義輸出目錄
        self.output_dirs = {
            'image_identify': self.base_output_dir / 'image_identify',
            'material_analysis': self.base_output_dir / 'material_analysis',
            'obj_models': self.base_output_dir / 'obj_models',
            'modified_obj': self.base_output_dir / 'modified_obj',
            'geometry_analysis': self.base_output_dir / 'geometry_analysis',
            'bootstrap_analysis': self.base_output_dir / 'bootstrap_analysis',
            'enhanced_analysis': self.base_output_dir / 'enhanced_analysis',
            'lca_results': self.base_output_dir / 'lca_results',
            'workflow_reports': self.base_output_dir / 'workflow_reports'
        }
        
        # 創建必要目錄
        self._create_directories()
    
    def _create_directories(self):
        """創建所有必要的輸出目錄"""
        for dir_name, dir_path in self.output_dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"確保目錄存在: {dir_path}")
    
    def phase_1_chair_recognition(self) -> Dict:
        """階段1: 椅子辨識"""
        logger.info("=" * 80)
        logger.info("🔍 階段1: 椅子辨識")
        logger.info("=" * 80)
        
        phase_start = time.time()
        
        try:
            # 檢查是否有待處理的圖片
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            input_images = []
            
            # 在當前目錄及子目錄尋找圖片
            for ext in image_extensions:
                input_images.extend(self.base_output_dir.glob(f"*{ext}"))
                input_images.extend(self.base_output_dir.glob(f"**/*{ext}"))
            
            if not input_images:
                logger.warning("未找到待處理的圖片文件")
                return {'success': False, 'error': 'No input images found'}
            
            logger.info(f"找到 {len(input_images)} 張圖片待處理")
            
            # 執行椅子辨識
            cmd = [yolo, "identifyChair.py"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                phase_time = time.time() - phase_start
                self.timing_records['chair_recognition'] = phase_time
                
                # 統計結果
                pass_dir = self.output_dirs['image_identify'] / 'identify_pass'
                fail_dir = self.output_dirs['image_identify'] / 'identify_fail'
                
                # 取得所有jpg檔案，但排除original_img資料夾
                if pass_dir.exists():
                    all_pass_files = list(pass_dir.rglob('*.jpg'))
                    pass_files = [f for f in all_pass_files if 'original_img' not in f.parts]
                    pass_count = len(pass_files)
                else:
                    pass_count = 0
                
                fail_count = len(list(fail_dir.rglob('*.jpg'))) if fail_dir.exists() else 0
                
                self.results['chair_recognition'] = {
                    'success': True,
                    'total_images': len(input_images),
                    'pass_count': pass_count,
                    'fail_count': fail_count,
                    'success_rate': pass_count / len(input_images) * 100 if len(input_images) > 0 else 0
                }

                
                logger.info(f"✅ 椅子辨識完成 (耗時: {phase_time:.2f}秒)")
                logger.info(f"   成功: {pass_count}, 失敗: {fail_count}")
                
                return self.results['chair_recognition']
            else:
                logger.error(f"椅子辨識失敗: {result.stderr}")
                return {'success': False, 'error': result.stderr}
                
        except Exception as e:
            logger.error(f"椅子辨識異常: {e}")
            return {'success': False, 'error': str(e)}
    
    def phase_2_material_detection(self) -> Dict:
        """階段2: 材質檢測"""
        logger.info("=" * 80)
        logger.info("🎨 階段2: 材質檢測")
        logger.info("=" * 80)
        
        phase_start = time.time()
        
        try:
            # 檢查是否有成功辨識的椅子
            pass_dir = self.output_dirs['image_identify'] / 'identify_pass'
            if not pass_dir.exists() or not list(pass_dir.rglob('*.jpg')):
                logger.warning("沒有成功辨識的椅子圖片")
                return {'success': False, 'error': 'No identified chairs'}
            
            # 執行材質檢測
            cmd = [mmsegmentation, "materialDetection.py"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                phase_time = time.time() - phase_start
                self.timing_records['material_detection'] = phase_time
                
                # 讀取材質分析結果
                material_results = self._load_material_results()
                
                self.results['material_detection'] = {
                    'success': True,
                    'analyzed_chairs': len(material_results),
                    'material_summary': self._summarize_materials(material_results)
                }
                
                logger.info(f"✅ 材質檢測完成 (耗時: {phase_time:.2f}秒)")
                
                return self.results['material_detection']
            else:
                logger.error(f"材質檢測失敗: {result.stderr}")
                return {'success': False, 'error': result.stderr}
                
        except Exception as e:
            logger.error(f"材質檢測異常: {e}")
            return {'success': False, 'error': str(e)}
    
    def phase_3_3d_generation(self) -> Dict:
        """階段3: 3D模型生成"""
        logger.info("=" * 80)
        logger.info("🎯 階段3: 3D模型生成")
        logger.info("=" * 80)
        
        phase_start = time.time()
        
        try:
            # 執行3D生成
            cmd = [trellis, "trellisAutoGeneration.py"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                phase_time = time.time() - phase_start
                self.timing_records['3d_generation'] = phase_time
                
                # 統計生成的模型
                model_dir = self.base_output_dir / '3d_models'
                glb_files = list(model_dir.rglob('*.glb')) if model_dir.exists() else []
                
                self.results['3d_generation'] = {
                    'success': True,
                    'generated_models': len(glb_files)
                }
                
                logger.info(f"✅ 3D模型生成完成 (耗時: {phase_time:.2f}秒)")
                logger.info(f"   生成模型數: {len(glb_files)}")
                
                return self.results['3d_generation']
            else:
                logger.error(f"3D模型生成失敗: {result.stderr}")
                return {'success': False, 'error': result.stderr}
                
        except Exception as e:
            logger.error(f"3D模型生成異常: {e}")
            return {'success': False, 'error': str(e)}
    
    def phase_4_model_processing(self) -> Dict:
        """階段4: 3D模型處理"""
        logger.info("=" * 80)
        logger.info("🔧 階段4: 3D模型處理")
        logger.info("=" * 80)
        
        phase_start = time.time()
        
        try:
            # 檢查是否有3D模型
            model_dir = self.base_output_dir / '3d_models'
            if not model_dir.exists() or not list(model_dir.rglob('*.glb')):
                logger.warning("沒有3D模型可處理")
                return {'success': False, 'error': 'No 3D models found'}
            
            # 執行工作流程整合
            cmd = ["python3.10", "workflowIntegration.py"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                phase_time = time.time() - phase_start
                self.timing_records['model_processing'] = phase_time
                
                self.results['model_processing'] = {
                    'success': True,
                    'obj_converted': len(list(self.output_dirs['obj_models'].rglob('*.obj'))),
                    'models_modified': len(list(self.output_dirs['modified_obj'].rglob('*.obj')))
                }
                
                logger.info(f"✅ 3D模型處理完成 (耗時: {phase_time:.2f}秒)")
                
                return self.results['model_processing']
            else:
                logger.error(f"3D模型處理失敗: {result.stderr}")
                return {'success': False, 'error': result.stderr}
                
        except Exception as e:
            logger.error(f"3D模型處理異常: {e}")
            return {'success': False, 'error': str(e)}
    
    # def phase_5_lca_analysis(self) -> Dict:
    #     """階段5: LCA分析 - 使用 OpenLCA"""
    #     logger.info("=" * 80)
    #     logger.info("♻️  階段5: LCA (生命週期評估) 分析")
    #     logger.info("=" * 80)
        
    #     phase_start = time.time()
        
    #     try:
    #         # 導入 OpenLCA 客戶端
    #         from openlca import OpenLCAClient, perform_quick_lca_assessment
            
    #         # 收集分析所需的數據
    #         chair_data_list = self._collect_chair_data()
            
    #         if not chair_data_list:
    #             logger.warning("沒有足夠的數據進行LCA分析")
    #             return {'success': False, 'error': 'Insufficient data for LCA'}
            
    #         lca_results = []
            
    #         # 嘗試連接 OpenLCA 服務器
    #         try:
    #             client = OpenLCAClient(host="localhost", port=8080)
    #             use_openlca_server = True
    #             logger.info("✅ 成功連接到 OpenLCA 服務器")
    #         except ConnectionError:
    #             logger.warning("⚠️  無法連接到 OpenLCA 服務器，使用快速評估模式")
    #             use_openlca_server = False
    #             client = None
            
    #         # 對每個椅子進行 LCA 分析
    #         for i, chair_data in enumerate(chair_data_list):
    #             logger.info(f"分析椅子 {i+1}/{len(chair_data_list)}...")
                
    #             try:
    #                 if use_openlca_server and client:
    #                     # 使用 OpenLCA 服務器進行完整 LCA
    #                     lca_result = client.perform_simplified_lca(chair_data)
    #                 else:
    #                     # 使用快速評估
    #                     lca_result = perform_quick_lca_assessment({
    #                         'weight': chair_data.get('geometry', {}).get('estimated_weight', 5.0),
    #                         'materials': chair_data.get('material_composition', {})
    #                     })
                    
    #                 lca_result['chair_id'] = chair_data.get('id', f'chair_{i+1}')
    #                 lca_results.append(lca_result)
                    
    #             except Exception as e:
    #                 logger.error(f"椅子 {i+1} LCA 分析失敗: {e}")
    #                 continue
            
    #         # 關閉 OpenLCA 客戶端
    #         if client:
    #             client.close()
            
    #         # 保存 LCA 結果
    #         self._save_lca_results(lca_results)
            
    #         phase_time = time.time() - phase_start
    #         self.timing_records['lca_analysis'] = phase_time
            
    #         # 計算總體統計
    #         if lca_results:
    #             total_carbon = sum(r.get('total_carbon_footprint', 0) for r in lca_results)
    #             avg_carbon = total_carbon / len(lca_results)
                
    #             self.results['lca_analysis'] = {
    #                 'success': True,
    #                 'analyzed_chairs': len(lca_results),
    #                 'total_carbon_footprint': total_carbon,
    #                 'average_carbon_footprint': avg_carbon,
    #                 'using_openlca_server': use_openlca_server
    #             }
                
    #             logger.info(f"✅ LCA 分析完成 (耗時: {phase_time:.2f}秒)")
    #             logger.info(f"   分析椅子數: {len(lca_results)}")
    #             logger.info(f"   平均碳足跡: {avg_carbon:.2f} kg CO2e")
                
    #             return self.results['lca_analysis']
    #         else:
    #             return {'success': False, 'error': 'No LCA results generated'}
                
    #     except Exception as e:
    #         logger.error(f"LCA 分析異常: {e}")
    #         return {'success': False, 'error': str(e)}
    def phase_5_lca_analysis(self) -> Dict:
        """階段5: LCA分析 - 使用 OpenLCA"""
        logger.info("=" * 80)
        logger.info("♻️  階段5: LCA (生命週期評估) 分析")
        logger.info("=" * 80)
        
        phase_start = time.time()
        
        try:
           # 收集分析所需的數據
            chair_data_list = self._collect_chair_data()
            
            if not chair_data_list:
                logger.warning("沒有足夠的數據進行LCA分析")
                return {'success': False, 'error': 'Insufficient data for LCA'}
            
            lca_results = []
            
            # 嘗試連接 OpenLCA 服務器
            try:
                client = olca.Client(port=8080)  # 正確的客戶端初始化方式
                use_openlca_server = True
                logger.info("✅ 成功連接到 OpenLCA 服務器")
            except Exception:
                logger.warning("⚠️  無法連接到 OpenLCA 服務器，使用快速評估模式")
                use_openlca_server = False
                client = None
            
            # 對每個椅子進行 LCA 分析
            for i, chair_data in enumerate(chair_data_list):
                logger.info(f"分析椅子 {i+1}/{len(chair_data_list)}...")
                
                try:
                    if use_openlca_server and client:
                        # 使用 OpenLCA 服務器進行完整 LCA
                        lca_result = self._perform_openlca_calculation(client, chair_data)
                    else:
                        # 使用快速評估（自定義函數）
                        lca_result = self._perform_quick_lca_assessment({
                            'weight': chair_data.get('geometry', {}).get('estimated_weight', 5.0),
                            'materials': chair_data.get('material_composition', {})
                        })
                    
                    lca_result['chair_id'] = chair_data.get('id', f'chair_{i+1}')
                    lca_results.append(lca_result)
                    
                except Exception as e:
                    logger.error(f"椅子 {i+1} LCA 分析失敗: {e}")
                    continue
            
            # 關閉 OpenLCA 客戶端
            if client:
                client.close()
            
            # 保存 LCA 結果
            self._save_lca_results(lca_results)
            
            phase_time = time.time() - phase_start
            self.timing_records['lca_analysis'] = phase_time
            
            # 計算總體統計
            if lca_results:
                total_carbon = sum(r.get('total_carbon_footprint', 0) for r in lca_results)
                avg_carbon = total_carbon / len(lca_results)
                
                self.results['lca_analysis'] = {
                    'success': True,
                    'analyzed_chairs': len(lca_results),
                    'total_carbon_footprint': total_carbon,
                    'average_carbon_footprint': avg_carbon,
                    'using_openlca_server': use_openlca_server
                }
                
                logger.info(f"✅ LCA 分析完成 (耗時: {phase_time:.2f}秒)")
                logger.info(f"   分析椅子數: {len(lca_results)}")
                logger.info(f"   平均碳足跡: {avg_carbon:.2f} kg CO2e")
                
                return self.results['lca_analysis']
            else:
                return {'success': False, 'error': 'No LCA results generated'}
                
        except Exception as e:
            logger.error(f"LCA 分析異常: {e}")
            return {'success': False, 'error': str(e)}

    def _perform_openlca_calculation(self, client, chair_data):
        """使用 OpenLCA 進行實際 LCA 計算"""
        try:
            # 創建計算設定
            setup = olca.CalculationSetup()
            setup.calculation_type = olca.CalculationType.UPSTREAM_ANALYSIS
            
            # 假設您已經有產品系統和影響評估方法的 ID
            # 這些需要根據您的實際資料庫內容調整
            setup.product_system = olca.ref(
                olca.ProductSystem,
                'your-product-system-id'  # 替換為實際的產品系統 ID
            )
            
            setup.impact_method = olca.ref(
                olca.ImpactMethod,
                'your-impact-method-id'  # 替換為實際的影響評估方法 ID
            )
            
            setup.amount = chair_data.get('geometry', {}).get('estimated_weight', 1.0)
            
            # 執行計算
            result = client.calculate(setup)
            
            # 處理結果
            if result and hasattr(result, 'total_impacts'):
                carbon_footprint = 0
                for impact in result.total_impacts:
                    if 'carbon' in impact.impact_category.name.lower() or 'gwp' in impact.impact_category.name.lower():
                        carbon_footprint = impact.value
                        break
                
                return {
                    'total_carbon_footprint': carbon_footprint,
                    'calculation_method': 'openlca_server',
                    'raw_result': result
                }
            else:
                return {
                    'total_carbon_footprint': 0,
                    'calculation_method': 'openlca_server',
                    'error': 'No impact results returned'
                }
                
        except Exception as e:
            logger.error(f"OpenLCA 計算失敗: {e}")
            return {
                'total_carbon_footprint': 0,
                'calculation_method': 'openlca_server',
                'error': str(e)
            }

    def _perform_quick_lca_assessment(self, data):
        """快速 LCA 評估（當無法連接到 OpenLCA 服務器時使用）"""
        try:
            weight = data.get('weight', 5.0)
            materials = data.get('materials', {})
            
            # 簡化的碳足跡計算（基於材料和重量的估算）
            carbon_factors = {
                'steel': 2.5,      # kg CO2e per kg
                'plastic': 3.0,    # kg CO2e per kg
                'wood': 0.5,       # kg CO2e per kg
                'aluminum': 8.0,   # kg CO2e per kg
                'fabric': 5.0      # kg CO2e per kg
            }
            
            total_carbon = 0
            for material, percentage in materials.items():
                factor = carbon_factors.get(material.lower(), 2.0)  # 預設值
                material_weight = weight * (percentage / 100)
                total_carbon += material_weight * factor
            
            return {
                'total_carbon_footprint': total_carbon,
                'calculation_method': 'quick_assessment',
                'materials_breakdown': materials
            }
            
        except Exception as e:
            logger.error(f"快速評估失敗: {e}")
            return {
                'total_carbon_footprint': 0,
                'calculation_method': 'quick_assessment',
                'error': str(e)
            }

    def _collect_chair_data(self) -> List[Dict]:
        """收集椅子數據用於 LCA 分析"""
        chair_data_list = []
        
        # 收集材質數據
        material_results = self._load_material_results()
        
        # 收集幾何數據
        geometry_results = self._load_geometry_results()
        
        # 合併數據
        for chair_id in material_results:
            chair_data = {
                'id': chair_id,
                'material_composition': material_results.get(chair_id, {}).get('materials', {}),
                'geometry': geometry_results.get(chair_id, {
                    'estimated_weight': 5.0,  # 默認重量
                    'volume': 0.01,
                    'surface_area': 1.5
                })
            }
            
            # 確保材料百分比總和為100
            total_percentage = sum(chair_data['material_composition'].values())
            if total_percentage > 0 and total_percentage != 100:
                factor = 100 / total_percentage
                for material in chair_data['material_composition']:
                    chair_data['material_composition'][material] *= factor
            
            chair_data_list.append(chair_data)
        
        return chair_data_list
    
    def _load_material_results(self) -> Dict:
        """載入材質分析結果"""
        material_results = {}
        
        # 查找材質分析報告
        material_report_path = self.output_dirs['material_analysis'] / 'chair_material_statistics.json'
        
        if material_report_path.exists():
            try:
                with open(material_report_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for chair_id, chair_data in data.items():
                    # 提取平均材質組成
                    avg_materials = chair_data.get('average_material_composition', {})
                    
                    # 過濾並標準化材質名稱
                    standardized_materials = {}
                    material_mapping = {
                        'wood': 'wood',
                        'plastic': 'plastic',
                        'metal': 'metal',
                        'fabric': 'fabric',
                        'leather': 'leather',
                        'glass': 'glass',
                        'ceramic': 'ceramic',
                        'paper': 'paper',
                        'foam': 'foam'
                    }
                    
                    for material, percentage in avg_materials.items():
                        std_material = material_mapping.get(material, 'other')
                        if std_material in standardized_materials:
                            standardized_materials[std_material] += percentage
                        else:
                            standardized_materials[std_material] = percentage
                    
                    material_results[chair_id] = {
                        'materials': standardized_materials,
                        'primary_material': chair_data.get('primary_material', 'unknown')
                    }
                    
            except Exception as e:
                logger.error(f"載入材質結果失敗: {e}")
        
        return material_results
    
    def _load_geometry_results(self) -> Dict:
        """載入幾何分析結果"""
        geometry_results = {}
        
        # 查找幾何分析報告
        geometry_report_path = self.output_dirs['geometry_analysis'] / 'geometry_analysis_report.json'
        
        if geometry_report_path.exists():
            try:
                with open(geometry_report_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for result in data:
                    if 'processed_analysis' in result:
                        file_name = result['processed_analysis']['file_name']
                        # 從文件名提取椅子ID
                        chair_id = file_name.split('_')[0] if '_' in file_name else 'chair'
                        
                        geometry_results[chair_id] = {
                            'volume': result['processed_analysis']['geometric_analysis']['bounding_box_volume'],
                            'vertices': result['processed_analysis']['basic_statistics']['vertices_count'],
                            'faces': result['processed_analysis']['basic_statistics']['faces_count'],
                            'estimated_weight': self._estimate_weight(
                                result['processed_analysis']['geometric_analysis']['bounding_box_volume']
                            )
                        }
                        
            except Exception as e:
                logger.error(f"載入幾何結果失敗: {e}")
        
        return geometry_results
    
    def _estimate_weight(self, volume: float) -> float:
        """根據體積估算重量"""
        # 假設平均密度為 500 kg/m³ (混合材料)
        density = 500
        # 體積通常很小，需要適當縮放
        estimated_weight = volume * density * 0.001  # 轉換為合理的重量範圍
        
        # 限制在合理範圍內 (2-15 kg)
        return max(2.0, min(15.0, estimated_weight))
    
    def _save_lca_results(self, lca_results: List[Dict]):
        """保存 LCA 分析結果"""
        # 保存詳細結果
        detailed_path = self.output_dirs['lca_results'] / 'lca_detailed_results.json'
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(lca_results, f, indent=2, ensure_ascii=False)
        
        # 保存摘要報告
        summary_path = self.output_dirs['lca_results'] / 'lca_summary_report.json'
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_chairs_analyzed': len(lca_results),
            'total_carbon_footprint': sum(r.get('total_carbon_footprint', 0) for r in lca_results),
            'average_carbon_footprint': sum(r.get('total_carbon_footprint', 0) for r in lca_results) / len(lca_results) if lca_results else 0,
            'carbon_range': {
                'min': min(r.get('total_carbon_footprint', 0) for r in lca_results) if lca_results else 0,
                'max': max(r.get('total_carbon_footprint', 0) for r in lca_results) if lca_results else 0
            },
            'breakdown': self._calculate_lca_breakdown(lca_results)
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成可視化報告
        self._generate_lca_visualization(lca_results)
        
        logger.info(f"LCA 結果已保存到: {self.output_dirs['lca_results']}")
    
    def _calculate_lca_breakdown(self, lca_results: List[Dict]) -> Dict:
        """計算 LCA 各階段的平均貢獻"""
        if not lca_results:
            return {}
        
        breakdown = {
            'materials': 0,
            'manufacturing': 0,
            'transportation': 0,
            'use_phase': 0,
            'end_of_life': 0
        }
        
        for result in lca_results:
            if 'breakdown' in result:
                for phase, value in result['breakdown'].items():
                    if phase in breakdown:
                        breakdown[phase] += value
        
        # 計算平均值
        num_results = len(lca_results)
        for phase in breakdown:
            breakdown[phase] = breakdown[phase] / num_results if num_results > 0 else 0
        
        return breakdown
    
    def _generate_lca_visualization(self, lca_results: List[Dict]):
        """生成 LCA 可視化圖表"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 創建圖表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 碳足跡分布直方圖
            carbon_footprints = [r.get('total_carbon_footprint', 0) for r in lca_results]
            ax1.hist(carbon_footprints, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax1.set_xlabel('Carbon Footprint (kg CO2e)')
            ax1.set_ylabel('Number of Chairs')
            ax1.set_title('Distribution of Carbon Footprints')
            ax1.grid(True, alpha=0.3)
            
            # 2. 生命週期階段貢獻餅圖
            breakdown = self._calculate_lca_breakdown(lca_results)
            if breakdown:
                labels = list(breakdown.keys())
                values = list(breakdown.values())
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
                
                ax2.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Average Lifecycle Stage Contributions')
            
            # 3. 材料類型 vs 碳足跡
            material_carbon = {}
            for result in lca_results:
                if 'carbon_footprint' in result and 'material_emissions' in result['carbon_footprint']:
                    for material, data in result['carbon_footprint']['material_emissions'].items():
                        if material not in material_carbon:
                            material_carbon[material] = []
                        material_carbon[material].append(data['emissions_kg_co2e'])
            
            if material_carbon:
                materials = list(material_carbon.keys())
                avg_emissions = [np.mean(material_carbon[m]) for m in materials]
                
                ax3.bar(materials, avg_emissions, color='skyblue', edgecolor='navy')
                ax3.set_xlabel('Material Type')
                ax3.set_ylabel('Average Emissions (kg CO2e)')
                ax3.set_title('Average Carbon Emissions by Material')
                ax3.tick_params(axis='x', rotation=45)
            
            # 4. 累積碳足跡
            sorted_footprints = sorted(carbon_footprints)
            cumulative = np.cumsum(sorted_footprints)
            
            ax4.plot(range(1, len(sorted_footprints) + 1), cumulative, 'b-', linewidth=2)
            ax4.fill_between(range(1, len(sorted_footprints) + 1), cumulative, alpha=0.3)
            ax4.set_xlabel('Number of Chairs')
            ax4.set_ylabel('Cumulative Carbon Footprint (kg CO2e)')
            ax4.set_title('Cumulative Carbon Footprint')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存圖表
            chart_path = self.output_dirs['lca_results'] / 'lca_analysis_charts.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"LCA 可視化圖表已保存: {chart_path}")
            
        except Exception as e:
            logger.error(f"生成 LCA 可視化失敗: {e}")
    
    def _summarize_materials(self, material_results: Dict) -> Dict:
        """總結材料分析結果"""
        all_materials = {}
        
        for chair_data in material_results.values():
            materials = chair_data.get('materials', {})
            for material, percentage in materials.items():
                if material not in all_materials:
                    all_materials[material] = []
                all_materials[material].append(percentage)
        
        # 計算平均值
        summary = {}
        for material, percentages in all_materials.items():
            summary[material] = {
                'average': np.mean(percentages),
                'min': np.min(percentages),
                'max': np.max(percentages),
                'occurrences': len(percentages)
            }
        
        return summary
    
    def generate_comprehensive_report(self):
        """生成綜合分析報告"""
        logger.info("📊 生成綜合分析報告...")
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_execution_time': time.time() - self.start_time,
                'phases_completed': len(self.results)
            },
            'phase_results': self.results,
            'timing_breakdown': self.timing_records,
            'summary': {
                'total_images_processed': self.results.get('chair_recognition', {}).get('total_images', 0),
                'chairs_identified': self.results.get('chair_recognition', {}).get('pass_count', 0),
                'materials_analyzed': self.results.get('material_detection', {}).get('analyzed_chairs', 0),
                '3d_models_generated': self.results.get('3d_generation', {}).get('generated_models', 0),
                'lca_assessments': self.results.get('lca_analysis', {}).get('analyzed_chairs', 0),
                'total_carbon_footprint': self.results.get('lca_analysis', {}).get('total_carbon_footprint', 0)
            }
        }
        
        # 保存報告
        report_path = self.output_dirs['workflow_reports'] / f'comprehensive_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成 Markdown 報告
        self._generate_markdown_report(report, report_path.with_suffix('.md'))
        
        logger.info(f"綜合報告已保存: {report_path}")
        
        return report_path
    
    def _generate_markdown_report(self, report: Dict, output_path: Path):
        """生成 Markdown 格式的報告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 椅子分析綜合報告\n\n")
            f.write(f"生成時間: {report['metadata']['timestamp']}\n")
            f.write(f"總執行時間: {report['metadata']['total_execution_time']:.2f} 秒\n\n")
            
            f.write("## 執行摘要\n\n")
            summary = report['summary']
            f.write(f"- 處理圖片數: {summary['total_images_processed']}\n")
            f.write(f"- 識別椅子數: {summary['chairs_identified']}\n")
            f.write(f"- 材質分析數: {summary['materials_analyzed']}\n")
            f.write(f"- 生成3D模型: {summary['3d_models_generated']}\n")
            f.write(f"- LCA評估數: {summary['lca_assessments']}\n")
            f.write(f"- **總碳足跡: {summary['total_carbon_footprint']:.2f} kg CO2e**\n\n")
            
            f.write("## 各階段結果\n\n")
            for phase, results in report['phase_results'].items():
                f.write(f"### {phase.replace('_', ' ').title()}\n")
                if results.get('success'):
                    f.write("✅ 成功\n")
                    for key, value in results.items():
                        if key != 'success' and key != 'error':
                            f.write(f"- {key}: {value}\n")
                else:
                    f.write("❌ 失敗\n")
                    f.write(f"- 錯誤: {results.get('error', 'Unknown error')}\n")
                f.write("\n")
            
            f.write("## 時間分析\n\n")
            for phase, duration in report['timing_breakdown'].items():
                f.write(f"- {phase}: {duration:.2f} 秒\n")
    
    def run_complete_workflow(self):
        """執行完整的工作流程"""
        logger.info("🚀 開始執行椅子分析完整工作流程")
        logger.info("=" * 100)
        
        success = True
        
        # 執行各階段
        phases = [
            ("椅子辨識", self.phase_1_chair_recognition),
            ("材質檢測", self.phase_2_material_detection),
            ("3D模型生成", self.phase_3_3d_generation),
            ("3D模型處理", self.phase_4_model_processing),
            ("LCA分析", self.phase_5_lca_analysis)
        ]
        
        for phase_name, phase_func in phases:
            result = phase_func()
            if not result.get('success', False):
                logger.warning(f"⚠️  {phase_name}階段失敗，繼續執行後續流程")
                success = False
        
        # 生成綜合報告
        report_path = self.generate_comprehensive_report()
        
        total_time = time.time() - self.start_time
        
        # 最終總結
        logger.info("=" * 100)
        logger.info("📊 FINAL WORKFLOW SUMMARY")
        logger.info("=" * 100)
        
        if success:
            logger.info(f"🎉 All processes completed successfully! Total time: {total_time:.2f}s")
        else:
            logger.info(f"⚠️  Some processes failed. Total time: {total_time:.2f}s")
        
        # 輸出結果統計
        logger.info("📊 Processing Results Summary:")
        for phase, result in self.results.items():
            status = "✅ Success" if result.get('success') else "❌ Failed"
            time_taken = self.timing_records.get(phase, 0)
            logger.info(f"   - {phase.replace('_', ' ').title()}: {status} ({time_taken:.2f}s)")
        
        logger.info(f"   - Total Execution Time: {total_time:.2f}s")
        
        # 輸出目錄信息
        logger.info("📁 Output Directories:")
        for dir_name, dir_path in self.output_dirs.items():
            if dir_path.exists():
                file_count = len(list(dir_path.rglob('*.*')))
                logger.info(f"   - {dir_name}: {file_count} files")
        
        logger.info(f"📋 Comprehensive Report: {report_path}")
        
        return success


def main():
    """主函數"""
    # 創建工作流程實例
    workflow = ChairAnalysisWorkflow()
    
    # 執行完整工作流程
    success = workflow.run_complete_workflow()
    
    # 返回狀態碼
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())