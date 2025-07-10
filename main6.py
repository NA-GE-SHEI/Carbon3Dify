#!/usr/bin/env python3

import subprocess
import os
import sys
import time
import logging
import json
import numpy as np
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import re

load_dotenv(r"./.env", override=True)

yolo = os.path.expanduser(os.getenv("YOLO_PYTHON"))
mmsegmentation = os.path.expanduser(os.getenv("MM_PYTHON"))
trellis = os.path.expanduser(os.getenv("TRELLIS_PYTHON"))

log_path = r"./auto.log"
log = logging.getLogger()
handlers = RotatingFileHandler(log_path, "a", 1024*1024*5, 3, "utf-8")
log.addHandler(handlers)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s[%(levelname)s]%(funcName)s: %(message)s")
handlers.setFormatter(formatter)

logger = logging.getLogger("**main**")

try:
    from openLCA import OpenLCACalculator
    OPENLCA_AVAILABLE = True
except ImportError:
    OPENLCA_AVAILABLE = False
    logger.warning("OpenLCA模組未找到，將使用內建LCA分析")

def check_3d_models_generated():
    """檢查是否有3D模型生成完成（支援chair和table）"""
    models_dir = Path("./3d_models")
    
    # 支援的家具類型
    furniture_types = ["Chair", "Table"]
    total_glb_files = 0
    found_furniture = []
    
    for furniture_type in furniture_types:
        furniture_dir = models_dir / furniture_type
        
        if furniture_dir.exists():
            logger.info(f"找到{furniture_type}目錄，檢查生成完成狀態...")
            
            # 檢查是否有對應的generation目錄
            generation_dirs = [d for d in furniture_dir.iterdir() 
                              if d.is_dir() and d.name.startswith(f'{furniture_type}_generation_')]
            
            if generation_dirs:
                # 檢查是否有GLB文件
                furniture_glb_count = 0
                for gen_dir in generation_dirs:
                    glb_files = list(gen_dir.glob("*.glb"))
                    furniture_glb_count += len(glb_files)
                
                total_glb_files += furniture_glb_count
                if furniture_glb_count > 0:
                    found_furniture.append(furniture_type)
                    logger.info(f"✅ {furniture_type}: 找到 {len(generation_dirs)} 個生成變體目錄，共 {furniture_glb_count} 個GLB文件")
            else:
                logger.info(f"⏳ {furniture_type}: 尚未找到{furniture_type}_generation_*目錄，等待TRELLIS生成完成...")
        else:
            logger.info(f"⏳ {furniture_type}: 目錄尚未創建，等待TRELLIS生成完成...")
    
    if found_furniture:
        logger.info(f"✅ 發現已完成的家具類型: {', '.join(found_furniture)}，共 {total_glb_files} 個GLB文件")
        return True
    else:
        logger.info("⏳ 尚未發現任何完成的3D模型生成...")
        return False

def get_predicted_weight_from_workflow() -> Optional[float]:
    """從工作流程報告中獲取預測重量"""
    try:
        workflow_reports_dir = Path("./workflow_reports")
        if not workflow_reports_dir.exists():
            return None
            
        # 查找最新的工作流程報告
        report_files = list(workflow_reports_dir.glob("workflow_report_*.json"))
        if not report_files:
            return None
            
        latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_report, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
        
        # 方法1: 從 final_predicted_weight 讀取
        if 'final_predicted_weight' in workflow_data and workflow_data['final_predicted_weight'] is not None:
            return float(workflow_data['final_predicted_weight'])
        
        # 方法2: 從 predicted_weights 讀取
        if 'predicted_weights' in workflow_data:
            weights = workflow_data['predicted_weights']
            if 'step_5' in weights and weights['step_5'] is not None:
                return float(weights['step_5'])
            
            # 取最後一個非空的預測重量
            for step_key in reversed(list(weights.keys())):
                if weights[step_key] is not None:
                    return float(weights[step_key])
        
        # 方法3: 從 step_results 讀取
        if 'step_results' in workflow_data:
            for step_num in ['5', '4', '3']:
                if step_num in workflow_data['step_results']:
                    step_result = workflow_data['step_results'][step_num]
                    if 'predicted_weight' in step_result and step_result['predicted_weight'] is not None:
                        return float(step_result['predicted_weight'])
        
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ 從工作流程報告獲取預測重量失敗: {e}")
        return None

def calculate_average_from_results() -> Optional[float]:
    """從結果目錄計算平均預測重量"""
    try:
        predictions = []
        
        # 檢查 enhanced_rf_result 目錄
        for i in range(1, 11):
            result_dir = Path(f"./enhanced_rf_result/result_{i}")
            result_file = result_dir / "detailed_analysis_results.txt"
            
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 修改正則表達式以支援家具類型
                    match = re.search(r'新(?:椅子|桌子|家具)預測重量[：:]\s*(\d+\.?\d*)\s*kg', content)
                    if match:
                        weight = float(match.group(1))
                        predictions.append(weight)
                        logger.info(f"  ✅ result_{i}: {weight:.2f} kg")
                        
                except Exception as e:
                    logger.warning(f"  ⚠️ 讀取 result_{i} 失敗: {e}")
        
        # 檢查 enhanced_analysis 目錄
        if not predictions:
            enhanced_analysis_dir = Path("./enhanced_analysis")
            if enhanced_analysis_dir.exists():
                result_files = list(enhanced_analysis_dir.glob("**/detailed_analysis_results.txt"))
                for result_file in result_files:
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        match = re.search(r'新(?:椅子|桌子|家具)預測重量[：:]\s*(\d+\.?\d*)\s*kg', content)
                        if match:
                            weight = float(match.group(1))
                            predictions.append(weight)
                            logger.info(f"  ✅ enhanced_analysis: {weight:.2f} kg")
                            
                    except Exception as e:
                        logger.warning(f"  ⚠️ 讀取enhanced_analysis結果失敗: {e}")
        
        if predictions:
            avg_weight = sum(predictions) / len(predictions)
            logger.info(f"📊 平均預測重量: {avg_weight:.2f} kg (基於 {len(predictions)} 次預測)")
            return avg_weight
        
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ 計算平均預測重量失敗: {e}")
        return None

def collect_furniture_data_for_lca() -> List[Dict]:
    """收集家具數據用於LCA分析 - 支援chair和table"""
    logger.info("🔍 收集家具數據用於LCA分析...")
    
    # 預設重量 - 根據家具類型設定不同的預設值
    furniture_default_weights = {
        'Chair': 4.2,
        'Table': 8.5  # 桌子通常比椅子重
    }
    
    predicted_weight = None
    weight_source = 'default'
    
    try:
        # 1. 優先從工作流程報告獲取預測重量
        predicted_weight = get_predicted_weight_from_workflow()
        if predicted_weight is not None:
            weight_source = 'workflow_report'
            logger.info(f"✅ 從工作流程報告獲取預測重量: {predicted_weight:.2f} kg")
        else:
            # 2. 如果沒有找到，計算平均值
            predicted_weight = calculate_average_from_results()
            if predicted_weight is not None:
                weight_source = 'averaged_predictions'
                logger.info(f"✅ 從結果文件計算平均重量: {predicted_weight:.2f} kg")
        
        furniture_data = []
        
        # 從3D模型分析收集數據
        models_dir = Path("./3d_models")
        furniture_types = ["Chair", "Table"]
        
        for furniture_type in furniture_types:
            furniture_dir = models_dir / furniture_type
            default_weight = furniture_default_weights[furniture_type]
            
            # 如果沒有預測重量，使用該家具類型的預設重量
            current_weight = predicted_weight if predicted_weight is not None else default_weight
            current_weight_source = weight_source if predicted_weight is not None else 'default'
            
            # 驗證預測重量的合理性（根據家具類型調整範圍）
            if furniture_type == 'Chair' and (current_weight <= 0 or current_weight > 50):
                logger.warning(f"⚠️ 椅子預測重量 {current_weight:.2f} kg 超出合理範圍，使用預設值")
                current_weight = default_weight
                current_weight_source = 'default_fallback'
            elif furniture_type == 'Table' and (current_weight <= 0 or current_weight > 100):
                logger.warning(f"⚠️ 桌子預測重量 {current_weight:.2f} kg 超出合理範圍，使用預設值")
                current_weight = default_weight
                current_weight_source = 'default_fallback'
            
            logger.info(f"🎯 {furniture_type}最終使用的預測重量: {current_weight:.2f} kg (來源: {current_weight_source})")
            
            if furniture_dir.exists():
                logger.info(f"🏗️ 發現{furniture_type}的3D模型，正在收集幾何數據...")
                generation_dirs = [d for d in furniture_dir.iterdir()
                                 if d.is_dir() and d.name.startswith(f'{furniture_type}_generation_')]
                
                for i, gen_dir in enumerate(generation_dirs[:3]):
                    glb_files = list(gen_dir.glob("*.glb"))
                    if glb_files:
                        furniture_id = f"{furniture_type}_{gen_dir.name}"
                        
                        # 查找對應的obj文件目錄
                        obj_dir = Path("./obj_models") / f"{furniture_type.lower()}_{i+1}"
                        
                        # 根據家具類型設定不同的材料分析參數
                        if furniture_type == 'Chair':
                            material_analysis = {
                                'wood_percentage': 90,
                                'has_material_detection': False,
                                'default_material': 'wood'
                            }
                            surface_area = 450
                            thickness = 3.0
                            geometry_volume = 3000
                        else:  # Table
                            material_analysis = {
                                'wood_percentage': 85,
                                'has_material_detection': False,
                                'default_material': 'wood'
                            }
                            surface_area = 1200  # 桌面面積
                            thickness = 2.5  # 桌面厚度
                            geometry_volume = 8000  # 桌子通常體積更大
                        
                        furniture_item = {
                            'furniture_id': furniture_id,
                            'furniture_type': furniture_type,
                            '3d_model_path': str(glb_files[0]),
                            'obj_output_dir': str(obj_dir),
                            'model_count': len(glb_files),
                            'material_analysis': material_analysis,
                            'surface_area': surface_area,
                            'thickness': thickness,
                            'geometry_analysis': {
                                'bounding_box_volume': geometry_volume
                            },
                            'estimated_weight': current_weight,
                            'wood_type': 'generic_wood',
                            'weight_source': current_weight_source
                        }
                        furniture_data.append(furniture_item)
                        logger.info(f" ✅ 收集{furniture_type}3D模型數據: {furniture_id} (重量: {current_weight:.2f} kg)")
        
        # 如果沒有數據，創建預設配置
        if len(furniture_data) == 0:
            logger.info("⚠️ 未找到現有數據，使用預測重量創建預設家具數據...")
            
            # 為每種家具類型創建預設配置
            for i, furniture_type in enumerate(furniture_types):
                default_weight = furniture_default_weights[furniture_type]
                current_weight = predicted_weight if predicted_weight is not None else default_weight
                current_weight_source = weight_source if predicted_weight is not None else 'default'
                
                if furniture_type == 'Chair':
                    material_analysis = {
                        'wood_percentage': 90,
                        'has_material_detection': False,
                        'default_material': 'wood'
                    }
                    surface_area = 450
                    thickness = 3.0
                    geometry_volume = 3000
                else:  # Table
                    material_analysis = {
                        'wood_percentage': 85,
                        'has_material_detection': False,
                        'default_material': 'wood'
                    }
                    surface_area = 1200
                    thickness = 2.5
                    geometry_volume = 8000
                
                default_furniture = {
                    'furniture_id': f'Predicted_{furniture_type}_001',
                    'furniture_type': furniture_type,
                    'obj_output_dir': str(Path("./obj_models") / f"{furniture_type.lower()}_{i+1}"),
                    'material_analysis': material_analysis,
                    'surface_area': surface_area,
                    'thickness': thickness,
                    'geometry_analysis': {
                        'bounding_box_volume': geometry_volume
                    },
                    'estimated_weight': current_weight,
                    'wood_type': 'generic_wood',
                    'weight_source': current_weight_source,
                    'source': 'predicted_configuration'
                }
                furniture_data.append(default_furniture)
                logger.info(f" ✅ 創建預測{furniture_type}配置 (重量: {current_weight:.2f} kg)")
        
        logger.info(f"📋 總共收集到 {len(furniture_data)} 個家具的數據用於LCA分析")
        return furniture_data
        
    except Exception as e:
        logger.error(f"❌ 收集家具數據時發生錯誤: {e}")
        # 返回備用配置
        backup_data = []
        furniture_types = ["Chair", "Table"]
        for furniture_type in furniture_types:
            default_weight = furniture_default_weights[furniture_type]
            current_weight = predicted_weight if predicted_weight is not None else default_weight
            current_weight_source = weight_source if predicted_weight is not None else 'default'
            
            backup_item = {
                'furniture_id': f'Fallback_{furniture_type}',
                'furniture_type': furniture_type,
                'obj_output_dir': str(Path("./obj_models") / f"{furniture_type.lower()}_1"),
                'material_analysis': {
                    'wood_percentage': 90 if furniture_type == 'Chair' else 85,
                    'has_material_detection': False,
                    'default_material': 'wood'
                },
                'surface_area': 450 if furniture_type == 'Chair' else 1200,
                'thickness': 3.0 if furniture_type == 'Chair' else 2.5,
                'geometry_analysis': {
                    'bounding_box_volume': 3000 if furniture_type == 'Chair' else 8000
                },
                'estimated_weight': current_weight,
                'wood_type': 'generic_wood',
                'weight_source': current_weight_source,
                'source': 'fallback_configuration'
            }
            backup_data.append(backup_item)
        
        return backup_data
    
def calculate_furniture_lca_fixed(furniture_data: Dict) -> Dict:
    """修復版的家具LCA計算函數 - 支援不同家具類型"""
    try:
        furniture_id = furniture_data.get('furniture_id', 'Unknown')
        furniture_type = furniture_data.get('furniture_type', 'Unknown')
        estimated_weight = furniture_data.get('estimated_weight', 4.2)
        weight_source = furniture_data.get('weight_source', 'default')
        
        logger.info(f"🌳 開始LCA分析: {furniture_id} ({furniture_type}, 重量: {estimated_weight:.2f} kg, 來源: {weight_source})")
        
        # 1. 確定材料成分（預設為木頭）
        material_analysis = furniture_data.get('material_analysis', {})
        wood_percentage = material_analysis.get('wood_percentage', 90) / 100
        metal_percentage = 1 - wood_percentage
        wood_type = furniture_data.get('wood_type', 'generic_wood')
        
        # 2. 計算體積和重量
        geometry_analysis = furniture_data.get('geometry_analysis', {})
        estimated_volume = geometry_analysis.get('bounding_box_volume', 2500)
        
        # 木材密度數據
        wood_densities = {
            'oak': 0.75,
            'pine': 0.52,
            'birch': 0.65,
            'maple': 0.70,
            'beech': 0.72,
            'generic_wood': 0.65
        }
        
        wood_density = wood_densities.get(wood_type, 0.65)
        
        # 質量分配 - 使用動態重量
        wood_mass = estimated_weight * wood_percentage
        metal_mass = estimated_weight * metal_percentage
        
        # 3. 碳足跡係數 - 根據家具類型調整
        wood_carbon_factors = {
            'oak': 0.9,
            'pine': 0.95,
            'birch': 0.85,
            'maple': 0.88,
            'beech': 0.87,
            'generic_wood': 0.90
        }
        
        wood_carbon_factor = wood_carbon_factors.get(wood_type, 0.90)
        metal_carbon_factor = 2.8
        
        # 根據家具類型調整碳排放係數
        if furniture_type == 'Table':
            # 桌子通常需要更多的加工和更穩固的結構
            wood_carbon_factor *= 1.1
            metal_carbon_factor *= 1.05
        
        # 4. 生命週期階段配置
        life_cycle_stages = {
            'material_extraction': 0.25,
            'production': 0.45,
            'transport': 0.10,
            'use_phase': 0.15,
            'end_of_life': 0.05
        }
        
        # 5. 計算各階段碳排放
        wood_carbon = wood_mass * wood_carbon_factor
        metal_carbon = metal_mass * metal_carbon_factor
        total_material_carbon = wood_carbon + metal_carbon
        
        # 各階段碳排放計算
        stage_emissions = {}
        for stage, factor in life_cycle_stages.items():
            stage_emissions[stage] = total_material_carbon * factor
        
        total_carbon_footprint = sum(stage_emissions.values())
        
        # 6. 可持續性評分計算 - 根據家具類型調整
        base_sustainability = 85 if furniture_type == 'Chair' else 80  # 桌子稍微複雜一些
        
        sustainability_score = (
            wood_percentage * base_sustainability +
            max(0, (100 - total_carbon_footprint * 15)) * 0.3 +
            75 * 0.2
        )
        
        sustainability_score = max(0, min(100, sustainability_score))
        
        # 7. 等級評定
        if sustainability_score >= 80:
            sustainability_grade = "優秀 (Excellent)"
        elif sustainability_score >= 70:
            sustainability_grade = "良好 (Good)"
        elif sustainability_score >= 60:
            sustainability_grade = "一般 (Fair)"
        else:
            sustainability_grade = "需改進 (Needs Improvement)"
        
        carbon_efficiency = total_carbon_footprint / estimated_weight
        
        if carbon_efficiency <= 1.0:
            carbon_grade = "A+"
        elif carbon_efficiency <= 1.5:
            carbon_grade = "A"
        elif carbon_efficiency <= 2.0:
            carbon_grade = "B"
        else:
            carbon_grade = "C"
        
        # 8. 環境影響評估
        water_usage = 'Low' if wood_percentage > 0.8 else 'Medium'
        biodiversity_impact = 'Positive' if wood_percentage > 0.9 else 'Neutral'
        
        # 9. 生成改進建議 - 根據家具類型
        recommendations = []
        if wood_percentage < 0.8:
            recommendations.append(f"建議增加{furniture_type}的木材使用比例以提高可持續性")
        if total_carbon_footprint > 3.5:
            recommendations.append("考慮使用本地木材供應以減少運輸碳足跡")
        if sustainability_score < 70:
            recommendations.append("考慮使用認證可持續木材（如FSC認證）")
        if furniture_type == 'Table' and total_carbon_footprint > 6.0:
            recommendations.append("考慮優化桌子結構設計以減少材料使用")
        if not recommendations:
            recommendations.append(f"當前{furniture_type}設計已具有良好的環境表現")
        
        # 構建完整的LCA結果
        lca_result = {
            'furniture_id': furniture_id,
            'furniture_type': furniture_type,
            'analysis_timestamp': datetime.now().isoformat(),
            'weight_info': {
                'estimated_weight_kg': estimated_weight,
                'weight_source': weight_source,
                'weight_used_in_calculation': estimated_weight
            },
            'materials': {
                'wood': wood_percentage,
                'metal': metal_percentage,
                'wood_type': wood_type,
                'wood_properties': {
                    'density': wood_density,
                    'carbon_factor': wood_carbon_factor
                }
            },
            'volume_mass': {
                'total_volume_cm3': estimated_volume,
                'total_mass_kg': estimated_weight,
                'wood_mass_kg': wood_mass,
                'metal_mass_kg': metal_mass,
                'density_g_cm3': estimated_weight * 1000 / estimated_volume
            },
            'life_cycle_impact': {
                'material_extraction_kg_co2': stage_emissions['material_extraction'],
                'production_kg_co2': stage_emissions['production'],
                'transport_kg_co2': stage_emissions['transport'],
                'use_phase_kg_co2': stage_emissions['use_phase'],
                'end_of_life_kg_co2': stage_emissions['end_of_life'],
                'total_life_cycle_kg_co2': total_carbon_footprint
            },
            'carbon_footprint': {
                'wood_carbon_kg_co2': wood_carbon,
                'metal_carbon_kg_co2': metal_carbon,
                'total_carbon_kg_co2': total_carbon_footprint,
                'carbon_efficiency_kg_co2_per_kg': carbon_efficiency,
                'carbon_grade': carbon_grade
            },
            'environmental_impact': {
                'renewable_score': wood_percentage * 10,
                'recycling_score': wood_percentage * 8.5 + metal_percentage * 9.5,
                'water_impact_score': 8 if water_usage == 'Low' else 5,
                'overall_environmental_score': (wood_percentage * 10 + 
                                              (8 if water_usage == 'Low' else 5) + 
                                              wood_percentage * 8.5) / 3
            },
            'sustainability_score': {
                'material_sustainability_score': wood_percentage * base_sustainability,
                'carbon_footprint_score': max(0, 100 - total_carbon_footprint * 10),
                'lifespan_score': 50,
                'overall_sustainability_score': sustainability_score,
                'sustainability_grade': sustainability_grade
            },
            'recommendations': recommendations
        }
        
        logger.info(f"✅ LCA分析完成: {furniture_id} ({furniture_type}) - 使用重量: {estimated_weight:.2f} kg - 碳足跡: {total_carbon_footprint:.2f} kg CO2e, 可持續性: {sustainability_score:.1f}/100")
        
        return lca_result
        
    except Exception as e:
        logger.error(f"❌ LCA分析失敗: {furniture_data.get('furniture_id', 'Unknown')} - {e}")
        return {
            'furniture_id': furniture_data.get('furniture_id', 'Unknown'),
            'furniture_type': furniture_data.get('furniture_type', 'Unknown'),
            'analysis_timestamp': datetime.now().isoformat(),
            'error': str(e),
            'success': False
        }

def run_enhanced_lca_analysis(furniture_data: List[Dict]) -> Dict:
    """運行增強版LCA分析 - 支援多種家具類型"""
    logger.info("♻️ 開始增強版LCA (生命週期評估) 分析")
    
    try:
        if not furniture_data:
            logger.warning("⚠️ 沒有家具數據可進行LCA分析")
            return {
                'success': False,
                'error': 'No furniture data available',
                'message': '沒有足夠的數據進行LCA分析'
            }
        
        logger.info(f"📊 正在分析 {len(furniture_data)} 個家具的生命週期評估...")
        
        # 統計家具類型
        furniture_types_count = {}
        for item in furniture_data:
            furniture_type = item.get('furniture_type', 'Unknown')
            furniture_types_count[furniture_type] = furniture_types_count.get(furniture_type, 0) + 1
        
        logger.info(f"家具類型分布: {dict(furniture_types_count)}")
        
        # 執行LCA分析
        lca_results = []
        successful_analyses = 0
        failed_analyses = 0
        
        for furniture_item in furniture_data:
            try:
                lca_result = calculate_furniture_lca_fixed(furniture_item)
                if lca_result.get('success', True):
                    lca_results.append(lca_result)
                    successful_analyses += 1
                    
                    # 保存每個家具的最終結果到JSON
                    try:
                        obj_output_dir = furniture_item.get('obj_output_dir', Path("./obj_models") / f"furniture_{successful_analyses}")
                        json_file = save_final_results_to_json(furniture_item, lca_result, obj_output_dir)
                        
                        if json_file:
                            logger.info(f"家具 {furniture_item.get('furniture_id', f'furniture_{successful_analyses}')} 的結果已保存到: {json_file}")
                    except Exception as json_error:
                        logger.error(f"❌ 保存家具 {furniture_item.get('furniture_id')} 的JSON結果時發生錯誤: {json_error}")
                        
                else:
                    failed_analyses += 1
                    logger.error(f" ❌ {furniture_item['furniture_id']}: LCA分析失敗")
            except Exception as e:
                failed_analyses += 1
                logger.error(f" ❌ {furniture_item['furniture_id']}: 分析異常 - {e}")
        
        if successful_analyses == 0:
            logger.error("❌ 所有家具LCA分析都失敗了")
            return {
                'success': False,
                'error': 'All LCA analyses failed',
                'message': '所有LCA分析都失敗了'
            }
        
        # 計算統計摘要
        carbon_footprints = [r['carbon_footprint']['total_carbon_kg_co2'] for r in lca_results]
        sustainability_scores = [r['sustainability_score']['overall_sustainability_score'] for r in lca_results]
        wood_percentages = [r['materials']['wood'] * 100 for r in lca_results]
        
        # 按家具類型分組統計
        type_stats = {}
        for result in lca_results:
            furniture_type = result.get('furniture_type', 'Unknown')
            if furniture_type not in type_stats:
                type_stats[furniture_type] = {
                    'count': 0,
                    'carbon_footprints': [],
                    'sustainability_scores': [],
                    'wood_percentages': []
                }
            type_stats[furniture_type]['count'] += 1
            type_stats[furniture_type]['carbon_footprints'].append(result['carbon_footprint']['total_carbon_kg_co2'])
            type_stats[furniture_type]['sustainability_scores'].append(result['sustainability_score']['overall_sustainability_score'])
            type_stats[furniture_type]['wood_percentages'].append(result['materials']['wood'] * 100)
        
        summary_stats = {
            'total_furniture': len(furniture_data),
            'successful_analysis': successful_analyses,
            'failed_analysis': failed_analyses,
            'furniture_types_analyzed': dict(furniture_types_count),
            'average_carbon_footprint': np.mean(carbon_footprints) if carbon_footprints else 0,
            'min_carbon_footprint': np.min(carbon_footprints) if carbon_footprints else 0,
            'max_carbon_footprint': np.max(carbon_footprints) if carbon_footprints else 0,
            'average_sustainability_score': np.mean(sustainability_scores) if sustainability_scores else 0,
            'min_sustainability_score': np.min(sustainability_scores) if sustainability_scores else 0,
            'max_sustainability_score': np.max(sustainability_scores) if sustainability_scores else 0,
            'average_wood_percentage': np.mean(wood_percentages) if wood_percentages else 0,
            'type_statistics': {},
            'material_distribution': {},
            'sustainability_grades': {}
        }
        
        # 計算各家具類型的統計
        for furniture_type, stats in type_stats.items():
            summary_stats['type_statistics'][furniture_type] = {
                'count': stats['count'],
                'avg_carbon': np.mean(stats['carbon_footprints']) if stats['carbon_footprints'] else 0,
                'avg_sustainability': np.mean(stats['sustainability_scores']) if stats['sustainability_scores'] else 0,
                'avg_wood_percentage': np.mean(stats['wood_percentages']) if stats['wood_percentages'] else 0
            }
        
        # 材料分布統計
        for result in lca_results:
            wood_ratio = result['materials']['wood']
            furniture_type = result.get('furniture_type', 'Unknown')
            if wood_ratio >= 0.9:
                category = f'{furniture_type} - High Wood Content (≥90%)'
            elif wood_ratio >= 0.7:
                category = f'{furniture_type} - Medium Wood Content (70-89%)'
            else:
                category = f'{furniture_type} - Low Wood Content (<70%)'
            
            summary_stats['material_distribution'][category] = \
                summary_stats['material_distribution'].get(category, 0) + 1
        
        # 可持續性等級分布
        for result in lca_results:
            grade = result['sustainability_score']['sustainability_grade']
            furniture_type = result.get('furniture_type', 'Unknown')
            type_grade = f'{furniture_type} - {grade}'
            summary_stats['sustainability_grades'][type_grade] = \
                summary_stats['sustainability_grades'].get(type_grade, 0) + 1
        
        # 生成建議
        recommendations = []
        avg_carbon = summary_stats['average_carbon_footprint']
        avg_sustainability = summary_stats['average_sustainability_score']
        avg_wood = summary_stats['average_wood_percentage']
        
        if avg_carbon > 4.0:  # 調整閾值以適應桌子較高的碳足跡
            recommendations.append("平均碳足跡偏高，建議優化生產工藝和選用更環保的木材")
        elif avg_carbon < 2.5:
            recommendations.append("碳足跡表現優秀，建議繼續保持當前的環保設計")
        
        if avg_sustainability < 70:
            recommendations.append("可持續性有提升空間，建議增加可持續認證木材的使用比例")
        elif avg_sustainability >= 80:
            recommendations.append("可持續性表現優秀，符合綠色設計標準")
        
        if avg_wood < 85:
            recommendations.append("建議提高木材使用比例以增強可持續性")
        
        # 針對不同家具類型的建議
        for furniture_type, stats in summary_stats['type_statistics'].items():
            if furniture_type == 'Table' and stats['avg_carbon'] > 6.0:
                recommendations.append(f"桌子的碳足跡較高，建議優化結構設計或使用更輕量化的材料")
            elif furniture_type == 'Chair' and stats['avg_carbon'] > 3.0:
                recommendations.append(f"椅子的碳足跡可以進一步優化，考慮減少金屬配件使用")
        
        recommendations.append("考慮使用FSC認證木材以進一步提升環境績效")
        
        # 保存詳細報告
        reports_dir = Path("./workflow_reports")
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON報告
        json_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'summary_statistics': summary_stats,
            'individual_results': lca_results,
            'recommendations': recommendations
        }
        
        json_file = reports_dir / f'furniture_lca_analysis_report_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        # 文字摘要報告
        txt_file = reports_dir / f'furniture_lca_analysis_summary_{timestamp}.txt'
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("家具生命週期評估 (LCA) 分析報告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析家具總數: {summary_stats['total_furniture']}\n")
            f.write(f"成功分析: {summary_stats['successful_analysis']}\n")
            f.write(f"分析失敗: {summary_stats['failed_analysis']}\n\n")
            
            f.write("家具類型分布:\n")
            f.write("-" * 40 + "\n")
            for furniture_type, count in summary_stats['furniture_types_analyzed'].items():
                f.write(f"{furniture_type}: {count} 個\n")
            f.write("\n")
            
            if summary_stats['successful_analysis'] > 0:
                f.write("整體統計結果:\n")
                f.write("-" * 40 + "\n")
                f.write(f"平均碳足跡: {summary_stats['average_carbon_footprint']:.2f} kg CO2e\n")
                f.write(f"平均可持續性評分: {summary_stats['average_sustainability_score']:.1f}/100\n")
                f.write(f"平均木材含量: {summary_stats['average_wood_percentage']:.1f}%\n\n")
                
                f.write("各家具類型統計:\n")
                f.write("-" * 40 + "\n")
                for furniture_type, stats in summary_stats['type_statistics'].items():
                    f.write(f"{furniture_type}:\n")
                    f.write(f"  數量: {stats['count']} 個\n")
                    f.write(f"  平均碳足跡: {stats['avg_carbon']:.2f} kg CO2e\n")
                    f.write(f"  平均可持續性: {stats['avg_sustainability']:.1f}/100\n")
                    f.write(f"  平均木材含量: {stats['avg_wood_percentage']:.1f}%\n\n")
                
                f.write("材料分布:\n")
                f.write("-" * 40 + "\n")
                for category, count in summary_stats['material_distribution'].items():
                    percentage = count / summary_stats['successful_analysis'] * 100
                    f.write(f"{category}: {count} 個 ({percentage:.1f}%)\n")
                
                f.write("\n可持續性等級分布:\n")
                f.write("-" * 40 + "\n")
                for grade, count in summary_stats['sustainability_grades'].items():
                    percentage = count / summary_stats['successful_analysis'] * 100
                    f.write(f"{grade}: {count} 個 ({percentage:.1f}%)\n")
                
                f.write("\n改進建議:\n")
                f.write("-" * 40 + "\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
        
        logger.info("✅ LCA分析完成")
        logger.info(f"📊 成功分析 {successful_analyses}/{len(furniture_data)} 個家具")
        if successful_analyses > 0:
            logger.info(f"🌱 平均可持續性評分: {summary_stats['average_sustainability_score']:.1f}/100")
            logger.info(f"♻️ 平均碳足跡: {summary_stats['average_carbon_footprint']:.2f} kg CO2e")
            logger.info(f"🌳 平均木材含量: {summary_stats['average_wood_percentage']:.1f}%")
            
            # 輸出各家具類型的統計
            for furniture_type, stats in summary_stats['type_statistics'].items():
                logger.info(f"📋 {furniture_type}: {stats['count']}個, 平均碳足跡: {stats['avg_carbon']:.2f} kg CO2e")
        
        logger.info(f"📄 詳細報告已保存: {json_file}")
        logger.info(f"📄 摘要報告已保存: {txt_file}")
        
        return {
            'success': True,
            'summary': summary_stats,
            'detailed_results': lca_results,
            'recommendations': recommendations,
            'report_files': {
                'json_report': str(json_file),
                'text_summary': str(txt_file)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ LCA分析過程中發生異常: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'LCA分析過程中發生錯誤'
        }

def run_openlca_analysis(furniture_data: List[Dict]) -> Dict:
    """使用OpenLCA進行專業LCA分析 - 支援多種家具類型"""
    try:
        logger.info("♻️ 開始OpenLCA專業LCA分析")
        
        # 初始化OpenLCA計算器
        calculator = OpenLCACalculator(port=8080)
        
        # 連接到OpenLCA
        if not calculator.connect():
            logger.error("❌ 無法連接到OpenLCA服務器")
            return {
                'success': False,
                'error': 'Cannot connect to OpenLCA',
                'message': '無法連接到OpenLCA服務器，請確保OpenLCA正在運行並開啟了IPC服務器'
            }
        
        # 執行批量計算
        logger.info(f"📊 開始批量計算 {len(furniture_data)} 個家具...")
        batch_results = calculator.batch_calculate_furniture(furniture_data)
        
        # 關閉連接
        calculator.close()
        
        if batch_results['success']:
            logger.info("✅ OpenLCA專業分析完成")
            logger.info(f"📊 成功分析 {batch_results['successful_analysis']}/{batch_results['total_furniture']} 個家具")
            logger.info(f"⏱️ 處理時間: {batch_results['processing_time']:.1f}秒")
            logger.info(f"📄 詳細結果已保存: {batch_results['results_file']}")
            
            # 轉換為與原有格式兼容的結果
            results_data = batch_results['results_data']
            
            # 計算統計摘要
            carbon_columns = [col for col in results_data.columns 
                            if 'climate change' in col.lower() or 'global warming' in col.lower()]
            
            if carbon_columns:
                carbon_values = results_data[carbon_columns[0]].dropna()
                avg_carbon = carbon_values.mean() if len(carbon_values) > 0 else 0
            else:
                avg_carbon = 0
            
            # 構建兼容的返回格式
            compatible_results = []
            for _, row in results_data.iterrows():
                furniture_result = {
                    'furniture_id': row.get('furniture_id', 'Unknown'),
                    'furniture_type': row.get('furniture_type', 'Unknown'),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'calculation_method': 'OpenLCA_Professional',
                    'materials': {
                        'wood_percentage': row.get('wood_percentage', 90) / 100 if 'wood_percentage' in row else 0.9,
                        'has_material_detection': True
                    },
                    'estimated_weight': row.get('estimated_weight', 4.0),
                    'carbon_footprint': {
                        'total_carbon_kg_co2': carbon_values.iloc[0] if len(carbon_values) > 0 else avg_carbon,
                        'carbon_efficiency_kg_co2_per_kg': (carbon_values.iloc[0] / row.get('estimated_weight', 4.0)) if len(carbon_values) > 0 else 0,
                        'source': 'OpenLCA_Database'
                    },
                    'sustainability_score': {
                        'overall_sustainability_score': 75,
                        'sustainability_grade': 'Good'
                    },
                    'openlca_results': row.to_dict(),
                    'success': True
                }
                compatible_results.append(furniture_result)
            
            summary_stats = {
                'total_furniture': batch_results['total_furniture'],
                'successful_analysis': batch_results['successful_analysis'],
                'failed_analysis': batch_results['failed_analysis'],
                'average_carbon_footprint': avg_carbon,
                'average_sustainability_score': 75,
                'calculation_method': 'OpenLCA_Professional',
                'processing_time': batch_results['processing_time']
            }
            
            return {
                'success': True,
                'summary': summary_stats,
                'detailed_results': compatible_results,
                'report_files': {
                    'csv_report': batch_results['results_file'],
                    'excel_report': batch_results['excel_file']
                },
                'openlca_raw_results': batch_results
            }
            
        else:
            logger.error("❌ OpenLCA批量分析失敗")
            logger.error(f"錯誤信息: {batch_results.get('error', 'Unknown error')}")
            return {
                'success': False,
                'error': batch_results.get('error', 'OpenLCA analysis failed'),
                'message': 'OpenLCA專業分析失敗'
            }
            
    except Exception as e:
        logger.error(f"❌ OpenLCA分析過程中發生異常: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'OpenLCA分析過程中發生錯誤'
        }

def run_workflow_integration(workflow_config=None):
    """運行工作流程整合腳本"""
    try:
        logger.info("=" * 60)
        logger.info("🚀 開始執行3D模型處理工作流程")
        logger.info("=" * 60)
        
        # 檢查工作流程整合腳本是否存在
        workflow_script = "workflowIntegration.py"
        if not os.path.exists(workflow_script):
            logger.error(f"❌ 找不到工作流程整合腳本: {workflow_script}")
            return False
        
        # 準備命令行參數
        cmd = [sys.executable, workflow_script]
        
        # 添加配置參數
        if workflow_config:
            if 'start_step' in workflow_config:
                cmd.extend(['--start_step', str(workflow_config['start_step'])])
            if 'end_step' in workflow_config:
                cmd.extend(['--end_step', str(workflow_config['end_step'])])
            if 'glb_dir' in workflow_config:
                cmd.extend(['--glb_dir', workflow_config['glb_dir']])
        
        logger.info(f"執行命令: {' '.join(cmd)}")
        
        # 執行工作流程
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            logger.info("✅ 3D模型處理工作流程執行成功")
            if result.stdout:
                logger.info("輸出信息:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
            return True
        else:
            logger.error("❌ 3D模型處理工作流程執行失敗")
            if result.stderr:
                logger.error("錯誤信息:")
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        logger.error(f"  {line}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 執行工作流程時發生異常: {e}")
        return False

def generate_comprehensive_report(workflow_results: Dict, lca_results: Dict):
    """生成綜合分析報告 - 支援多種家具類型"""
    logger.info("📊 生成綜合分析報告...")
    
    try:
        reports_dir = Path("./workflow_reports")
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"comprehensive_furniture_report_{timestamp}.json"
        
        # 處理 lca_results，移除不可序列化的 DataFrame
        lca_results_serializable = {}
        for key, value in lca_results.items():
            if hasattr(value, 'to_dict'):
                lca_results_serializable[key] = value.to_dict('records')
            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                lca_results_serializable[key] = str(value)
            else:
                lca_results_serializable[key] = value
        
        # 特別處理 detailed_results 中可能的 DataFrame
        if 'detailed_results' in lca_results_serializable:
            detailed_results = lca_results_serializable['detailed_results']
            if isinstance(detailed_results, list):
                for i, result in enumerate(detailed_results):
                    if isinstance(result, dict):
                        for result_key, result_value in result.items():
                            if hasattr(result_value, 'to_dict'):
                                detailed_results[i][result_key] = result_value.to_dict('records')
                            elif hasattr(result_value, '__dict__') and not isinstance(result_value, (str, int, float, bool, list, dict, type(None))):
                                detailed_results[i][result_key] = str(result_value)
        
        comprehensive_report = {
            'report_timestamp': datetime.now().isoformat(),
            'workflow_results': workflow_results,
            'lca_analysis': lca_results_serializable,
            'summary': {
                'workflow_success': workflow_results.get('success', False),
                'lca_success': lca_results.get('success', False),
                'total_furniture_analyzed': lca_results.get('summary', {}).get('successful_analysis', 0),
                'furniture_types_analyzed': lca_results.get('summary', {}).get('furniture_types_analyzed', {}),
                'average_carbon_footprint': lca_results.get('summary', {}).get('average_carbon_footprint', 0),
                'average_sustainability_score': lca_results.get('summary', {}).get('average_sustainability_score', 0),
                'average_wood_percentage': lca_results.get('summary', {}).get('average_wood_percentage', 0),
                'type_statistics': lca_results.get('summary', {}).get('type_statistics', {})
            },
            'recommendations': []
        }
        
        # 生成建議
        if lca_results.get('success', False):
            lca_summary = lca_results.get('summary', {})
            avg_carbon = lca_summary.get('average_carbon_footprint', 0)
            avg_sustainability = lca_summary.get('average_sustainability_score', 0)
            type_stats = lca_summary.get('type_statistics', {})
            
            if avg_carbon > 4.0:
                comprehensive_report['recommendations'].append(
                    "整體碳足跡偏高，建議優化材料選擇和生產工藝"
                )
            
            if avg_sustainability < 70:
                comprehensive_report['recommendations'].append(
                    "可持續性有提升空間，建議增加可持續材料使用比例"
                )
            
            if avg_sustainability >= 80:
                comprehensive_report['recommendations'].append(
                    "家具設計具有良好的環境表現，繼續保持"
                )
            
            # 針對不同家具類型的建議
            for furniture_type, stats in type_stats.items():
                if furniture_type == 'Table' and stats.get('avg_carbon', 0) > 6.0:
                    comprehensive_report['recommendations'].append(
                        f"桌子的碳足跡較高，建議優化結構設計或使用更輕量化的材料"
                    )
                elif furniture_type == 'Chair' and stats.get('avg_carbon', 0) > 3.5:
                    comprehensive_report['recommendations'].append(
                        f"椅子的碳足跡可以進一步優化，考慮減少金屬配件使用"
                    )
        
        # 保存報告，使用自定義序列化處理
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ 綜合報告已保存: {report_file}")
        return str(report_file)
        
    except Exception as e:
        logger.error(f"❌ 生成綜合報告時發生錯誤: {e}")
        logger.error(f"錯誤詳情: {type(e).__name__}: {str(e)}")
        return None
    
def save_final_results_to_json(furniture_data, lca_result, output_dir):
    """將最後預測重量和計算後的二氧化碳排放結果保存到JSON文件"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        furniture_id = furniture_data.get('furniture_id', 'unknown_furniture')
        furniture_type = furniture_data.get('furniture_type', 'Unknown')

        # 準備要保存的資料
        result_data = {
            'furniture_id': furniture_id,
            'furniture_type': furniture_type,
            'predicted_weight_kg': lca_result.get('weight_info', {}).get('estimated_weight_kg', None),
            'total_co2_eq_kg': lca_result.get('carbon_footprint', {}).get('total_carbon_kg_co2', None),
            'analysis_timestamp': datetime.now().isoformat(),
            'weight_source': lca_result.get('weight_info', {}).get('weight_source', 'unknown')
        }

        json_file = output_path / f'{furniture_id}_final_result.json'

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 最終結果已保存到: {json_file}")
        return str(json_file)

    except Exception as e:
        logger.error(f"❌ 保存JSON結果時發生錯誤: {e}")
        return None

def main():
    """主函數"""
    start = time.time()
    overall_success = True
    
    logger.info("🚀 開始執行增強版家具處理流程...")
    logger.info("支援家具類型: 椅子(Chair)、桌子(Table)")
    logger.info("流程包括: 家具辨識 → 材質檢測 → 3D模型生成 → 3D模型處理分析 → LCA生命週期評估")
    logger.info("=" * 80)
    
    # ======================== 階段1: 家具辨識 ========================
    try:
        logger.info("♻️ 階段1: 家具辨識 (椅子、桌子)")
        logger.info("=" * 80)
        
        identifyFurniture_result = subprocess.run([yolo, "identifyFurniture.py"], capture_output=False, text=True)
        
        if identifyFurniture_result.returncode == 0:
            logger.info("✅ 家具辨識完成")
        else:
            logger.error("❌ 家具辨識失敗")
            logger.error(f"錯誤信息: {identifyFurniture_result.stderr}")
            overall_success = False
            
    except Exception as e:
        logger.error(f"❌ 家具辨識階段異常: {e}")
        overall_success = False
    
    # ======================== 階段2: 材質檢測 ========================
    try:
        logger.info("♻️ 階段2: 材質檢測")
        logger.info("=" * 80)
        
        materialDetection_result = subprocess.run([mmsegmentation, "materialDetection.py"], capture_output=False, text=True)
        
        if materialDetection_result.returncode == 0:
            logger.info("✅ 材質檢測完成")
        else:
            logger.error("❌ 材質檢測失敗")
            logger.error(f"錯誤信息: {materialDetection_result.stderr}")
            overall_success = False
            
    except Exception as e:
        logger.error(f"❌ 材質檢測階段異常: {e}")
        overall_success = False
    
    # ======================== 階段3: 3D模型生成 ========================
    try:
        logger.info("♻️ 階段3: 3D模型生成")
        logger.info("=" * 80)
        
        trellis_result = subprocess.run([trellis, "trellisAutoGeneration.py"], capture_output=False, text=True)
        
        if trellis_result.returncode == 0:
            logger.info("✅ 3D模型生成完成")
        else:
            logger.error("❌ 3D模型生成失敗")
            logger.error(f"錯誤信息: {trellis_result.stderr}")
            overall_success = False
            
    except Exception as e:
        logger.error(f"❌ 3D模型生成階段異常: {e}")
        overall_success = False
    
    # ======================== 階段4: 3D模型處理工作流程 ========================
    workflow_success = False
    workflow_results = {'success': False}
    
    try:
        logger.info("♻️ 階段4: 3D模型處理工作流程")
        logger.info("=" * 80)
        
        # 檢查3D模型是否生成完成
        if check_3d_models_generated():
            logger.info("✅ 檢測到3D模型，開始執行後續處理工作流程...")
            
            # 配置工作流程參數
            workflow_config = {
                'start_step': 1,
                'end_step': 5,
                'glb_dir': './3d_models'
            }
            
            workflow_success = run_workflow_integration(workflow_config)
            workflow_results = {'success': workflow_success}
            
            if workflow_success:
                logger.info("✅ 3D模型處理工作流程執行成功")
            else:
                logger.error("❌ 3D模型處理工作流程執行失敗")
                overall_success = False
        else:
            logger.warning("⚠️ 未檢測到3D模型生成或生成不完整，跳過後續處理")
            logger.info("您可以稍後手動執行: python workflowIntegration.py")
            
    except Exception as e:
        logger.error(f"❌ 3D模型處理工作流程異常: {e}")
        overall_success = False
    
    # ======================== 階段5: LCA (生命週期評估) 分析 ========================
    lca_results = {'success': False}
    
    try:
        logger.info("♻️ 階段5: LCA (生命週期評估) 分析")
        logger.info("=" * 80)
        
        # 收集家具數據
        furniture_data = collect_furniture_data_for_lca()
        
        if furniture_data:
            logger.info(f"✅ 成功收集到 {len(furniture_data)} 個家具的數據")
            
            # 統計家具類型
            furniture_types = {}
            for item in furniture_data:
                furniture_type = item.get('furniture_type', 'Unknown')
                furniture_types[furniture_type] = furniture_types.get(furniture_type, 0) + 1
            logger.info(f"家具類型分布: {dict(furniture_types)}")
            
            if OPENLCA_AVAILABLE:
                logger.info("🔄 嘗試使用OpenLCA進行專業LCA分析...")
                lca_results = run_openlca_analysis(furniture_data)
                
                if not lca_results.get('success', False):
                    logger.warning("⚠️ OpenLCA分析失敗，使用備用方法...")
                    lca_results = run_enhanced_lca_analysis(furniture_data)
                else:
                    logger.info("✅ 成功使用OpenLCA完成專業LCA分析")
            else:
                logger.info("🔄 使用內建LCA分析...")
                lca_results = run_enhanced_lca_analysis(furniture_data)
            
            if lca_results['success']:
                summary = lca_results['summary']
                logger.info("✅ LCA分析完成")
                logger.info(f"📊 分析結果: 成功分析 {summary['successful_analysis']}/{summary['total_furniture']} 個家具")
                
                if summary['successful_analysis'] > 0:
                    logger.info(f"🌱 平均可持續性評分: {summary['average_sustainability_score']:.1f}/100")
                    logger.info(f"♻️ 平均碳足跡: {summary['average_carbon_footprint']:.2f} kg CO2e")
                    
                    # 安全地訪問 average_wood_percentage
                    wood_percentage = summary.get('average_wood_percentage', 0)
                    logger.info(f"🌳 平均木材含量: {wood_percentage:.1f}%")
                    
                    # 輸出各家具類型的統計
                    type_stats = summary.get('type_statistics', {})
                    for furniture_type, stats in type_stats.items():
                        logger.info(f"📋 {furniture_type}: {stats['count']}個, 平均碳足跡: {stats['avg_carbon']:.2f} kg CO2e")
                    
                    # 輸出JSON保存信息
                    logger.info("💾 最終結果JSON文件已保存到與obj文件相同的目錄中")
            else:
                logger.error("❌ LCA分析失敗")
                logger.error(f"錯誤信息: {lca_results.get('error', 'Unknown error')}")
                overall_success = False
        else:
            logger.warning("⚠️ 沒有足夠的數據進行LCA分析")
            logger.warning("⚠️ LCA分析階段失敗，繼續執行後續流程")
            
    except Exception as e:
        logger.error(f"❌ LCA分析階段異常: {e}")
        overall_success = False
    
    # ======================== 階段6: 生成綜合分析報告 ========================
    try:
        logger.info("📊 生成綜合分析報告...")
        comprehensive_report_file = generate_comprehensive_report(workflow_results, lca_results)
        if comprehensive_report_file:
            logger.info(f"綜合報告已保存: {comprehensive_report_file}")
    except Exception as e:
        logger.error(f"❌ 生成綜合報告時發生錯誤: {e}")

    # ======================== 階段7: 生成視覺化報告 ========================
    report_results = None
    try:
        logger.info("♻️ 階段7: 生成視覺化報告")
        logger.info("=" * 80)
        
        # 從創建的模組導入函數
        from report import generate_visual_report, print_final_summary
        
        # 生成視覺化報告
        report_results = generate_visual_report(workflow_results, lca_results, logger)
        
        if report_results:
            logger.info("✅ 視覺化報告生成成功")
            overall_success = True
        else:
            logger.error("❌ 視覺化報告生成失敗")
            
    except Exception as e:
        logger.error(f"❌ 視覺化報告生成階段異常: {e}")

    # ======================== 總結 ========================
    et = time.time() - start
    logger.info("=" * 80)
    
    if overall_success:
        logger.info("🎉 整個家具處理流程執行完成！")
        print(f"🎉 All furniture processing completed successfully! 總耗時: {et:.2f}秒")
    else:
        logger.warning("⚠️ 部分流程執行失敗，請檢查日誌")
        print(f"⚠️ Some processes failed. 總耗時: {et:.2f}秒")
    
    logger.info("=" * 80)
    logger.info("📊 處理結果摘要:")
    logger.info(f" - 家具辨識: {'✅' if 'identifyFurniture_result' in locals() and identifyFurniture_result.returncode == 0 else '❌'}")
    logger.info(f" - 材質檢測: {'✅' if 'materialDetection_result' in locals() and materialDetection_result.returncode == 0 else '❌'}")
    logger.info(f" - 3D模型生成: {'✅' if 'trellis_result' in locals() and trellis_result.returncode == 0 else '❌'}")
    logger.info(f" - 3D模型處理: {'✅' if workflow_success else '❌'}")
    logger.info(f" - LCA分析: {'✅' if lca_results.get('success', False) else '❌'}")
    logger.info(f" - 總執行時間: {et:.2f}秒")
    
    # 輸出結果目錄信息
    logger.info("\n📁 輸出目錄:")
    output_dirs = [
        "./image_identify",
        "./material_analysis", 
        "./3d_models",
        "./obj_models",
        "./modified_obj",
        "./geometry_analysis",
        "./bootstrap_analysis",
        "./enhanced_analysis",
        "./workflow_reports"
    ]
    
    for output_dir in output_dirs:
        if Path(output_dir).exists():
            file_count = len(list(Path(output_dir).rglob('*.*')))
            logger.info(f" - {output_dir}: {file_count} 個文件")
    
    logger.info("=" * 80)
    
    # 輸出LCA分析結果摘要
    if lca_results.get('success', False):
        logger.info("🌱 LCA分析結果摘要:")
        summary = lca_results['summary']
        logger.info(f" - 分析家具數量: {summary['successful_analysis']}/{summary['total_furniture']}")
        
        # 輸出家具類型分布
        furniture_types = summary.get('furniture_types_analyzed', {})
        if furniture_types:
            logger.info(" - 家具類型分布:")
            for furniture_type, count in furniture_types.items():
                logger.info(f"   {furniture_type}: {count} 個")
        
        if summary['successful_analysis'] > 0:
            logger.info(f" - 平均碳足跡: {summary['average_carbon_footprint']:.2f} kg CO2e")
            logger.info(f" - 平均可持續性評分: {summary['average_sustainability_score']:.1f}/100")
            
            # 安全地訪問 average_wood_percentage
            wood_percentage = summary.get('average_wood_percentage', 0)
            logger.info(f" - 平均木材含量: {wood_percentage:.1f}%")
            
            # 輸出各家具類型統計
            type_stats = summary.get('type_statistics', {})
            if type_stats:
                logger.info(" - 各家具類型統計:")
                for furniture_type, stats in type_stats.items():
                    logger.info(f"   {furniture_type}: 平均碳足跡 {stats['avg_carbon']:.2f} kg CO2e, 可持續性 {stats['avg_sustainability']:.1f}/100")
            
            if 'report_files' in lca_results:
                report_files = lca_results['report_files']
                if 'json_report' in report_files:
                    logger.info(f" - LCA詳細報告: {report_files['json_report']}")
                if 'text_summary' in report_files:
                    logger.info(f" - LCA摘要報告: {report_files['text_summary']}")
                
                # 處理 OpenLCA 結果的不同鍵名
                if 'csv_report' in report_files:
                    logger.info(f" - LCA CSV報告: {report_files['csv_report']}")
                if 'excel_report' in report_files:
                    logger.info(f" - LCA Excel報告: {report_files['excel_report']}")
            
            # 輸出JSON結果保存信息
            logger.info("💾 最終結果JSON文件已保存到與obj文件相同的目錄中")
            
    # 打印最終摘要
    if 'print_final_summary' in locals():
        print_final_summary(workflow_results, lca_results, report_results, logger)

if __name__ == '__main__':
    # 檢查所需腳本是否存在
    required_scripts = [
        "identifyFurniture.py",  # 修改為支援家具辨識的腳本名稱
        "materialDetection.py", 
        "trellisAutoGeneration.py",
        "workflowIntegration.py"
    ]
    
    missing_scripts = [script for script in required_scripts if not os.path.exists(script)]
    
    if missing_scripts:
        print(f"❌ 缺少必要的腳本文件: {missing_scripts}")
        logging.error(f"缺少必要的腳本文件: {missing_scripts}")
        
        # 如果只是 identifyFurniture.py 不存在，提示用戶可能需要重命名
        if "identifyFurniture.py" in missing_scripts and os.path.exists("identifyChair.py"):
            print("💡 提示: 發現 identifyChair.py，您可能需要:")
            print("   1. 重命名為 identifyFurniture.py，或")
            print("   2. 修改腳本以支援椅子和桌子的辨識")
        
        sys.exit(1)
    
    print("🚀 開始執行完整的家具處理流程...")
    print("支援家具類型: 椅子(Chair)、桌子(Table)")
    print("流程包括: 家具辨識 → 材質檢測 → 3D模型生成 → 3D模型處理分析 → LCA生命週期評估")
    
    main()