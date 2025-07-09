#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

load_dotenv(r"./.env", override=True)

yolo = os.path.expanduser(os.getenv("YOLO_PYTHON"))
mmsegmentation = os.path.expanduser(os.getenv("MM_PYTHON"))
trellis = os.path.expanduser(os.getenv("TRELLIS_PYTHON"))

# 設置日誌
log_path = r"./auto.log"
log = logging.getLogger()
handlers = RotatingFileHandler(log_path, "a", 1024*1024*5, 3, "utf-8")
log.addHandler(handlers)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s[%(levelname)s]%(funcName)s: %(message)s")
handlers.setFormatter(formatter)

# 設置主日誌記錄器
logger = logging.getLogger("**main**")

try:
    from openLCA import OpenLCACalculator
    OPENLCA_AVAILABLE = True
except ImportError:
    OPENLCA_AVAILABLE = False
    logger.warning("OpenLCA模組未找到，將使用內建LCA分析")

def check_3d_models_generated():
    """檢查是否有3D模型生成完成"""
    models_dir = Path("./3d_models")
    chair_dir = models_dir / "Chair"
    
    if not chair_dir.exists():
        logger.info("Chair目錄尚未創建，等待TRELLIS生成完成...")
        return False
    
    # 檢查是否有Chair_generation_*目錄
    generation_dirs = [d for d in chair_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('Chair_generation_')]
    
    if not generation_dirs:
        logger.info("尚未找到Chair_generation_*目錄，等待TRELLIS生成完成...")
        return False
    
    # 檢查是否有GLB文件
    total_glb_files = 0
    for gen_dir in generation_dirs:
        glb_files = list(gen_dir.glob("*.glb"))
        total_glb_files += len(glb_files)
    
    logger.info(f"找到 {len(generation_dirs)} 個生成變體目錄，共 {total_glb_files} 個GLB文件")
    
    # 如果有GLB文件就認為生成完成
    return total_glb_files > 0

def collect_chair_data_for_lca() -> List[Dict]:
    """收集椅子數據用於LCA分析 - 動態重量版"""
    logger.info("🔍 收集椅子數據用於LCA分析...")
    chairs_data = []
    
    # 預設重量
    default_weight = 4.2
    predicted_weight = default_weight
    weight_source = 'default'
    
    try:
        # 1. 從工作流程報告中讀取預測重量
        workflow_reports_dir = Path("./workflow_reports")
        if workflow_reports_dir.exists():
            # 查找最新的工作流程報告
            report_files = list(workflow_reports_dir.glob("workflow_report_*.json"))
            if report_files:
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_report, 'r', encoding='utf-8') as f:
                        workflow_data = json.load(f)
                    
                    # 方法1: 從 final_predicted_weight 讀取
                    if 'final_predicted_weight' in workflow_data and workflow_data['final_predicted_weight'] is not None:
                        predicted_weight = float(workflow_data['final_predicted_weight'])
                        weight_source = 'workflow_final_prediction'
                        logger.info(f"✅ 從工作流程報告獲取最終預測重量: {predicted_weight:.2f} kg")
                    
                    # 方法2: 從 predicted_weights 讀取
                    elif 'predicted_weights' in workflow_data:
                        weights = workflow_data['predicted_weights']
                        if 'step_5' in weights and weights['step_5'] is not None:
                            predicted_weight = float(weights['step_5'])
                            weight_source = 'workflow_step5_prediction'
                        elif weights:
                            # 取最後一個非空的預測重量
                            for step_key in reversed(list(weights.keys())):
                                if weights[step_key] is not None:
                                    predicted_weight = float(weights[step_key])
                                    weight_source = f'workflow_{step_key}_prediction'
                                    break
                        logger.info(f"✅ 從工作流程報告獲取預測重量: {predicted_weight:.2f} kg (來源: {weight_source})")
                    
                    # 方法3: 從 step_results 讀取
                    elif 'step_results' in workflow_data:
                        for step_num in ['5', '4', '3']:  # 優先順序
                            if step_num in workflow_data['step_results']:
                                step_result = workflow_data['step_results'][step_num]
                                if 'predicted_weight' in step_result and step_result['predicted_weight'] is not None:
                                    predicted_weight = float(step_result['predicted_weight'])
                                    weight_source = f'workflow_step{step_num}_prediction'
                                    logger.info(f"✅ 從步驟{step_num}獲取預測重量: {predicted_weight:.2f} kg")
                                    break
                    
                    # 驗證預測重量的合理性
                    if predicted_weight <= 0 or predicted_weight > 50:  # 椅子重量應該在合理範圍內
                        logger.warning(f"⚠️ 預測重量 {predicted_weight:.2f} kg 超出合理範圍，使用預設值")
                        predicted_weight = default_weight
                        weight_source = 'default_fallback'
                    
                except Exception as e:
                    logger.warning(f"⚠️ 讀取工作流程報告失敗: {e}")
                    logger.info(f"使用預設重量: {default_weight} kg")
        
        # 2. 如果沒有找到工作流程報告，嘗試從enhanced_analysis結果中讀取
        if predicted_weight == default_weight:
            enhanced_analysis_dir = Path("./enhanced_analysis")
            if enhanced_analysis_dir.exists():
                result_files = list(enhanced_analysis_dir.glob("**/detailed_analysis_results.txt"))
                if result_files:
                    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                    try:
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            import re
                            weight_pattern = r'新椅子預測重量[：:]\s*(\d+\.?\d*)\s*kg'
                            matches = re.findall(weight_pattern, content)
                            if matches:
                                # 取第一個匹配的結果（通常是正確的預測）
                                predicted_weight = float(matches[0])
                                weight_source = 'enhanced_analysis_file'
                                logger.info(f"✅ 從enhanced_analysis結果文件獲取重量: {predicted_weight:.2f} kg")
                    except Exception as e:
                        logger.warning(f"⚠️ 讀取enhanced_analysis結果失敗: {e}")
        
        # 記錄最終使用的重量
        logger.info(f"🎯 最終使用的預測重量: {predicted_weight:.2f} kg (來源: {weight_source})")
        
        # 3. 從3D模型分析收集數據，使用動態重量
        models_dir = Path("./3d_models")
        chair_dir = models_dir / "Chair"
        
        if chair_dir.exists():
            logger.info("🏗️ 發現3D模型，正在收集幾何數據...")
            generation_dirs = [d for d in chair_dir.iterdir()
                             if d.is_dir() and d.name.startswith('Chair_generation_')]
            
            for i, gen_dir in enumerate(generation_dirs[:3]):
                glb_files = list(gen_dir.glob("*.glb"))
                if glb_files:
                    chair_id = f"Chair_{gen_dir.name}"
                    
                    chair_data = {
                        'chair_id': chair_id,
                        '3d_model_path': str(glb_files[0]),
                        'model_count': len(glb_files),
                        'material_analysis': {
                            'wood_percentage': 90,
                            'has_material_detection': False,
                            'default_material': 'wood'
                        },
                        'seat_area': 450,
                        'seat_thickness': 3.0,
                        'geometry_analysis': {
                            'bounding_box_volume': 3000
                        },
                        'estimated_weight': predicted_weight,  # 使用動態預測重量
                        'wood_type': 'generic_wood',
                        'weight_source': weight_source
                    }
                    
                    chairs_data.append(chair_data)
                    logger.info(f" ✅ 收集3D模型數據: {chair_id} (重量: {predicted_weight:.2f} kg, 來源: {weight_source})")
        
        # 4. 如果沒有數據，使用預測重量創建預設配置
        if len(chairs_data) == 0:
            logger.info("⚠️ 未找到現有數據，使用預測重量創建預設椅子數據...")
            default_chair = {
                'chair_id': 'Predicted_Chair_001',
                'material_analysis': {
                    'wood_percentage': 90,
                    'has_material_detection': False,
                    'default_material': 'wood'
                },
                'seat_area': 450,
                'seat_thickness': 3.0,
                'geometry_analysis': {
                    'bounding_box_volume': 3000
                },
                'estimated_weight': predicted_weight,  # 使用動態預測重量
                'wood_type': 'generic_wood',
                'weight_source': weight_source,
                'source': 'predicted_configuration'
            }
            chairs_data.append(default_chair)
            logger.info(f" ✅ 創建預測椅子配置 (重量: {predicted_weight:.2f} kg, 來源: {weight_source})")
        
        logger.info(f"📋 總共收集到 {len(chairs_data)} 個椅子的數據用於LCA分析")
        
        # 保存收集的數據用於調試
        debug_file = Path("./workflow_reports") / f"lca_input_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        debug_file.parent.mkdir(exist_ok=True)
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(chairs_data, f, indent=2, ensure_ascii=False)
        logger.info(f"🔍 LCA輸入數據已保存至: {debug_file}")
        
        return chairs_data
        
    except Exception as e:
        logger.error(f"❌ 收集椅子數據時發生錯誤: {e}")
        # 如果出錯，至少返回一個基本的椅子配置
        fallback_data = [{
            'chair_id': 'Fallback_Chair',
            'material_analysis': {
                'wood_percentage': 90,
                'has_material_detection': False,
                'default_material': 'wood'
            },
            'seat_area': 450,
            'seat_thickness': 3.0,
            'geometry_analysis': {
                'bounding_box_volume': 3000
            },
            'estimated_weight': predicted_weight,  # 使用動態預測重量
            'wood_type': 'generic_wood',
            'weight_source': weight_source,
            'source': 'fallback_configuration'
        }]
        logger.info(f"🔄 使用備用椅子配置進行LCA分析 (重量: {predicted_weight:.2f} kg, 來源: {weight_source})")
        return fallback_data


def calculate_chair_lca_fixed(chair_data: Dict) -> Dict:
    """修復版的椅子LCA計算函數 - 使用動態重量"""
    try:
        chair_id = chair_data.get('chair_id', 'Unknown')
        estimated_weight = chair_data.get('estimated_weight', 4.2)  # 動態重量
        weight_source = chair_data.get('weight_source', 'default')
        
        logger.info(f"🌳 開始LCA分析: {chair_id} (重量: {estimated_weight:.2f} kg, 來源: {weight_source})")

        # 1. 確定材料成分（預設為木頭）
        material_analysis = chair_data.get('material_analysis', {})
        wood_percentage = material_analysis.get('wood_percentage', 90) / 100
        metal_percentage = 1 - wood_percentage
        wood_type = chair_data.get('wood_type', 'generic_wood')

        # 2. 計算體積和重量
        geometry_analysis = chair_data.get('geometry_analysis', {})
        estimated_volume = geometry_analysis.get('bounding_box_volume', 2500)  # cm³

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

        # 3. 碳足跡係數
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

        # 4. 生命週期階段配置
        life_cycle_stages = {
            'material_extraction': 0.25,
            'production': 0.45,
            'transport': 0.10,
            'use_phase': 0.15,
            'end_of_life': 0.05
        }

        # 5. 計算各階段碳排放 - 基於動態重量
        wood_carbon = wood_mass * wood_carbon_factor
        metal_carbon = metal_mass * metal_carbon_factor
        total_material_carbon = wood_carbon + metal_carbon

        # 各階段碳排放計算
        stage_emissions = {}
        for stage, factor in life_cycle_stages.items():
            stage_emissions[stage] = total_material_carbon * factor

        total_carbon_footprint = sum(stage_emissions.values())

        # 6. 可持續性評分計算
        sustainability_score = (
            wood_percentage * 85 +  # 木材可持續性基礎分
            max(0, (100 - total_carbon_footprint * 15)) * 0.3 +  # 碳效率分
            75 * 0.2  # 可回收性分
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

        # 9. 生成改進建議
        recommendations = []
        if wood_percentage < 0.8:
            recommendations.append("建議增加木材使用比例以提高可持續性")
        if total_carbon_footprint > 3.5:
            recommendations.append("考慮使用本地木材供應以減少運輸碳足跡")
        if sustainability_score < 70:
            recommendations.append("考慮使用認證可持續木材（如FSC認證）")
        if not recommendations:
            recommendations.append("當前設計已具有良好的環境表現")

        # 構建完整的LCA結果
        lca_result = {
            'chair_id': chair_id,
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
                'material_sustainability_score': wood_percentage * 85,
                'carbon_footprint_score': max(0, 100 - total_carbon_footprint * 10),
                'lifespan_score': 50,  # 假設25年壽命
                'overall_sustainability_score': sustainability_score,
                'sustainability_grade': sustainability_grade
            },
            'recommendations': recommendations
        }

        logger.info(f"✅ LCA分析完成: {chair_id} - 使用重量: {estimated_weight:.2f} kg - 碳足跡: {total_carbon_footprint:.2f} kg CO2e, 可持續性: {sustainability_score:.1f}/100")
        return lca_result

    except Exception as e:
        logger.error(f"❌ LCA分析失敗: {chair_data.get('chair_id', 'Unknown')} - {e}")
        # 返回基本的失敗結果而不是拋出異常
        return {
            'chair_id': chair_data.get('chair_id', 'Unknown'),
            'analysis_timestamp': datetime.now().isoformat(),
            'error': str(e),
            'success': False
        }

def run_enhanced_lca_analysis(chairs_data: List[Dict]) -> Dict:
    """運行增強版LCA分析 - 修復版"""
    logger.info("♻️ 開始增強版LCA (生命週期評估) 分析")
    
    try:
        if not chairs_data:
            logger.warning("⚠️ 沒有椅子數據可進行LCA分析")
            return {
                'success': False,
                'error': 'No chair data available',
                'message': '沒有足夠的數據進行LCA分析'
            }
        
        logger.info(f"📊 正在分析 {len(chairs_data)} 個椅子的生命週期評估...")
        
        # 執行LCA分析
        lca_results = []
        successful_analyses = 0
        failed_analyses = 0
        
        for chair_data in chairs_data:
            try:
                lca_result = calculate_chair_lca_fixed(chair_data)
                
                # 檢查分析是否成功
                if lca_result.get('success', True):  # 如果沒有success字段，預設為成功
                    lca_results.append(lca_result)
                    successful_analyses += 1
                else:
                    failed_analyses += 1
                    logger.error(f"  ❌ {chair_data['chair_id']}: LCA分析失敗")
                
            except Exception as e:
                failed_analyses += 1
                logger.error(f"  ❌ {chair_data['chair_id']}: 分析異常 - {e}")
        
        if successful_analyses == 0:
            logger.error("❌ 所有椅子LCA分析都失敗了")
            return {
                'success': False,
                'error': 'All LCA analyses failed',
                'message': '所有LCA分析都失敗了'
            }
        
        # 計算統計摘要
        carbon_footprints = [r['carbon_footprint']['total_carbon_kg_co2'] for r in lca_results]
        sustainability_scores = [r['sustainability_score']['overall_sustainability_score'] for r in lca_results]
        wood_percentages = [r['materials']['wood'] * 100 for r in lca_results]
        
        summary_stats = {
            'total_chairs': len(chairs_data),
            'successful_analysis': successful_analyses,
            'failed_analysis': failed_analyses,
            'average_carbon_footprint': np.mean(carbon_footprints) if carbon_footprints else 0,
            'min_carbon_footprint': np.min(carbon_footprints) if carbon_footprints else 0,
            'max_carbon_footprint': np.max(carbon_footprints) if carbon_footprints else 0,
            'average_sustainability_score': np.mean(sustainability_scores) if sustainability_scores else 0,
            'min_sustainability_score': np.min(sustainability_scores) if sustainability_scores else 0,
            'max_sustainability_score': np.max(sustainability_scores) if sustainability_scores else 0,
            'average_wood_percentage': np.mean(wood_percentages) if wood_percentages else 0,  # 確保這個字段存在
            'material_distribution': {},
            'sustainability_grades': {}
        }
        
        # 材料分布統計
        for result in lca_results:
            wood_ratio = result['materials']['wood']
            if wood_ratio >= 0.9:
                category = 'High Wood Content (≥90%)'
            elif wood_ratio >= 0.7:
                category = 'Medium Wood Content (70-89%)'
            else:
                category = 'Low Wood Content (<70%)'
            
            summary_stats['material_distribution'][category] = \
                summary_stats['material_distribution'].get(category, 0) + 1
        
        # 可持續性等級分布
        for result in lca_results:
            grade = result['sustainability_score']['sustainability_grade']
            summary_stats['sustainability_grades'][grade] = \
                summary_stats['sustainability_grades'].get(grade, 0) + 1
        
        # 生成建議
        recommendations = []
        avg_carbon = summary_stats['average_carbon_footprint']
        avg_sustainability = summary_stats['average_sustainability_score']
        avg_wood = summary_stats['average_wood_percentage']
        
        if avg_carbon > 3.5:
            recommendations.append("平均碳足跡偏高，建議優化生產工藝和選用更環保的木材")
        elif avg_carbon < 2.0:
            recommendations.append("碳足跡表現優秀，建議繼續保持當前的環保設計")
        
        if avg_sustainability < 70:
            recommendations.append("可持續性有提升空間，建議增加可持續認證木材的使用比例")
        elif avg_sustainability >= 80:
            recommendations.append("可持續性表現優秀，符合綠色設計標準")
        
        if avg_wood < 85:
            recommendations.append("建議提高木材使用比例以增強可持續性")
        
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
        
        json_file = reports_dir / f'lca_analysis_report_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        # 文字摘要報告
        txt_file = reports_dir / f'lca_analysis_summary_{timestamp}.txt'
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("椅子生命週期評估 (LCA) 分析報告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析椅子總數: {summary_stats['total_chairs']}\n")
            f.write(f"成功分析: {summary_stats['successful_analysis']}\n")
            f.write(f"分析失敗: {summary_stats['failed_analysis']}\n\n")
            
            if summary_stats['successful_analysis'] > 0:
                f.write("整體統計結果:\n")
                f.write("-" * 40 + "\n")
                f.write(f"平均碳足跡: {summary_stats['average_carbon_footprint']:.2f} kg CO2e\n")
                f.write(f"平均可持續性評分: {summary_stats['average_sustainability_score']:.1f}/100\n")
                f.write(f"平均木材含量: {summary_stats['average_wood_percentage']:.1f}%\n\n")
                
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
        logger.info(f"📊 成功分析 {successful_analyses}/{len(chairs_data)} 個椅子")
        if successful_analyses > 0:
            logger.info(f"🌱 平均可持續性評分: {summary_stats['average_sustainability_score']:.1f}/100")
            logger.info(f"♻️ 平均碳足跡: {summary_stats['average_carbon_footprint']:.2f} kg CO2e")
            logger.info(f"🌳 平均木材含量: {summary_stats['average_wood_percentage']:.1f}%")
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
def run_workflow_integration(workflow_config=None):
    """
    運行工作流程整合腳本
    
    Args:
        workflow_config: 工作流程配置字典，可選
    """
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
        result = subprocess.run(cmd, capture_output=False, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            logger.info("✅ 3D模型處理工作流程執行成功")
            logger.info("輸出信息:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
            return True
        else:
            logger.error("❌ 3D模型處理工作流程執行失敗")
            logger.error("錯誤信息:")
            for line in result.stderr.strip().split('\n'):
                if line.strip():
                    logger.error(f"  {line}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 執行工作流程時發生異常: {e}")
        return False

def generate_comprehensive_report(workflow_results: Dict, lca_results: Dict):
    """生成綜合分析報告 - 修復版"""
    logger.info("📊 生成綜合分析報告...")
    
    try:
        reports_dir = Path("./workflow_reports")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"comprehensive_report_{timestamp}.json"
        
        # 處理 lca_results，移除不可序列化的 DataFrame
        lca_results_serializable = {}
        for key, value in lca_results.items():
            if hasattr(value, 'to_dict'):  # 檢查是否為 DataFrame
                lca_results_serializable[key] = value.to_dict('records')
            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                # 其他不可序列化的對象
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
                            if hasattr(result_value, 'to_dict'):  # DataFrame
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
                'total_chairs_analyzed': lca_results.get('summary', {}).get('successful_analysis', 0),
                'average_carbon_footprint': lca_results.get('summary', {}).get('average_carbon_footprint', 0),
                'average_sustainability_score': lca_results.get('summary', {}).get('average_sustainability_score', 0),
                'average_wood_percentage': lca_results.get('summary', {}).get('average_wood_percentage', 0)  # 添加預設值
            },
            'recommendations': []
        }
        
        # 生成建議
        if lca_results.get('success', False):
            lca_summary = lca_results.get('summary', {})
            avg_carbon = lca_summary.get('average_carbon_footprint', 0)
            avg_sustainability = lca_summary.get('average_sustainability_score', 0)
            
            if avg_carbon > 3.0:
                comprehensive_report['recommendations'].append(
                    "碳足跡偏高，建議優化材料選擇和生產工藝"
                )
            
            if avg_sustainability < 70:
                comprehensive_report['recommendations'].append(
                    "可持續性有提升空間，建議增加可持續材料使用比例"
                )
            
            if avg_sustainability >= 80:
                comprehensive_report['recommendations'].append(
                    "椅子設計具有良好的環境表現，繼續保持"
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

def main():
    """主函數"""
    start = time.time()
    overall_success = True
    
    logger.info("🚀 開始執行增強版椅子處理流程...")
    logger.info("流程包括: 椅子辨識 → 材質檢測 → 3D模型生成 → 3D模型處理分析 → LCA生命週期評估")
    logger.info("=" * 80)
    
    # ======================== 階段1: 椅子辨識 ========================
    try:
        logger.info("♻️ 階段1: 椅子辨識")
        logger.info("=" * 80)
        
        identifyChair_result = subprocess.run([yolo, "identifyChair.py"], capture_output=False, text=True)
        
        if identifyChair_result.returncode == 0:
            logger.info("✅ 椅子辨識完成")
        else:
            logger.error("❌ 椅子辨識失敗")
            logger.error(f"錯誤信息: {identifyChair_result.stderr}")
            overall_success = False
            
    except Exception as e:
        logger.error(f"❌ 椅子辨識階段異常: {e}")
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
        
        # 收集椅子數據
        chairs_data = collect_chair_data_for_lca()
        
        if chairs_data:
            logger.info(f"✅ 成功收集到 {len(chairs_data)} 個椅子的數據")
            if OPENLCA_AVAILABLE:
                logger.info("🔄 嘗試使用OpenLCA進行專業LCA分析...")
                lca_results = run_openlca_analysis(chairs_data)
                
                if not lca_results.get('success', False):
                    logger.warning("⚠️ OpenLCA分析失敗，使用備用方法...")
                    lca_results = run_enhanced_lca_analysis(chairs_data)
                else:
                    logger.info("✅ 成功使用OpenLCA完成專業LCA分析")
            else:
                logger.info("🔄 使用內建LCA分析...")
                lca_results = run_enhanced_lca_analysis(chairs_data)
            
            if lca_results['success']:
                summary = lca_results['summary']
                logger.info("✅ LCA分析完成")
                logger.info(f"📊 分析結果: 成功分析 {summary['successful_analysis']}/{summary['total_chairs']} 個椅子")
                
                if summary['successful_analysis'] > 0:
                    logger.info(f"🌱 平均可持續性評分: {summary['average_sustainability_score']:.1f}/100")
                    logger.info(f"♻️ 平均碳足跡: {summary['average_carbon_footprint']:.2f} kg CO2e")
                    # 安全地訪問 average_wood_percentage
                    wood_percentage = summary.get('average_wood_percentage', 0)
                    logger.info(f"🌳 平均木材含量: {wood_percentage:.1f}%")
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
    
    # ======================== 總結 ========================
    et = time.time() - start
    
    logger.info("=" * 80)
    if overall_success:
        logger.info("🎉 整個處理流程執行完成！")
        print(f"🎉 All processes completed successfully! 總耗時: {et:.2f}秒")
    else:
        logger.warning("⚠️ 部分流程執行失敗，請檢查日誌")
        print(f"⚠️ Some processes failed. 總耗時: {et:.2f}秒")
    
    logger.info("=" * 80)
    logger.info("📊 處理結果摘要:")
    logger.info(f"   - 椅子辨識: {'✅' if 'identifyChair_result' in locals() and identifyChair_result.returncode == 0 else '❌'}")
    logger.info(f"   - 材質檢測: {'✅' if 'materialDetection_result' in locals() and materialDetection_result.returncode == 0 else '❌'}")
    logger.info(f"   - 3D模型生成: {'✅' if 'trellis_result' in locals() and trellis_result.returncode == 0 else '❌'}")
    logger.info(f"   - 3D模型處理: {'✅' if workflow_success else '❌'}")
    logger.info(f"   - LCA分析: {'✅' if lca_results.get('success', False) else '❌'}")
    logger.info(f"   - 總執行時間: {et:.2f}秒")
    
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
            logger.info(f"   - {output_dir}: {file_count} 個文件")
    
    logger.info("=" * 80)
    
    # 輸出LCA分析結果摘要
    if lca_results.get('success', False):
        logger.info("🌱 LCA分析結果摘要:")
        summary = lca_results['summary']
        logger.info(f"   - 分析椅子數量: {summary['successful_analysis']}/{summary['total_chairs']}")
        if summary['successful_analysis'] > 0:
            logger.info(f"   - 平均碳足跡: {summary['average_carbon_footprint']:.2f} kg CO2e")
            logger.info(f"   - 平均可持續性評分: {summary['average_sustainability_score']:.1f}/100")
            # 安全地訪問 average_wood_percentage
            wood_percentage = summary.get('average_wood_percentage', 0)
            logger.info(f"   - 平均木材含量: {wood_percentage:.1f}%")
        
        if 'report_files' in lca_results:
            report_files = lca_results['report_files']
            if 'json_report' in report_files:
                logger.info(f"   - LCA詳細報告: {report_files['json_report']}")
            if 'text_summary' in report_files:
                logger.info(f"   - LCA摘要報告: {report_files['text_summary']}")
            # 處理 OpenLCA 結果的不同鍵名
            if 'csv_report' in report_files:
                logger.info(f"   - LCA CSV報告: {report_files['csv_report']}")
            if 'excel_report' in report_files:
                logger.info(f"   - LCA Excel報告: {report_files['excel_report']}")

def run_openlca_analysis(chairs_data: List[Dict]) -> Dict:
    """使用OpenLCA進行專業LCA分析 - 修改版"""
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
        logger.info(f"📊 開始批量計算 {len(chairs_data)} 個椅子...")
        batch_results = calculator.batch_calculate_chairs(chairs_data)
        
        # 關閉連接
        calculator.close()
        
        if batch_results['success']:
            logger.info("✅ OpenLCA專業分析完成")
            logger.info(f"📊 成功分析 {batch_results['successful_analysis']}/{batch_results['total_chairs']} 個椅子")
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
                chair_result = {
                    'chair_id': row.get('chair_id', 'Unknown'),
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
                        'overall_sustainability_score': 75,  # 基於OpenLCA結果的簡化評分
                        'sustainability_grade': 'Good'
                    },
                    'openlca_results': row.to_dict(),
                    'success': True
                }
                compatible_results.append(chair_result)
            
            summary_stats = {
                'total_chairs': batch_results['total_chairs'],
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


if __name__ == '__main__':
    # 檢查所需腳本是否存在
    required_scripts = [
        "identifyChair.py",
        "materialDetection.py", 
        "trellisAutoGeneration.py",
        "workflowIntegration.py"
    ]
    
    missing_scripts = [script for script in required_scripts if not os.path.exists(script)]
    
    if missing_scripts:
        print(f"❌ 缺少必要的腳本文件: {missing_scripts}")
        logging.error(f"缺少必要的腳本文件: {missing_scripts}")
        sys.exit(1)
    
    print("🚀 開始執行完整的椅子處理流程（含修復版LCA分析）...")
    print("流程包括: 椅子辨識 → 材質檢測 → 3D模型生成 → 3D模型處理分析 → LCA生命週期評估")
    main()