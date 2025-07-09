#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced LCA (生命週期評估) 分析模組
解決"沒有足夠的數據進行LCA分析"的問題，預設材料為木頭
參考 openlca.py 並整合現有的材料分析

作者: AI Assistant
創建時間: 2025-07-08
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 設置日誌
logger = logging.getLogger(__name__)

class EnhancedLCAAnalyzer:
    """增強版LCA分析器 - 預設木頭材料"""
    
    def __init__(self, config: Dict = None):
        """
        初始化LCA分析器
        
        Args:
            config: LCA配置參數
        """
        self.config = config or self._default_config()
        self.material_database = self._initialize_material_database()
        self.analysis_results = []
        
    def _default_config(self) -> Dict:
        """預設LCA配置"""
        return {
            'default_material': 'wood',  # 預設材料為木頭
            'wood_types': {
                'oak': {'density': 0.75, 'carbon_factor': 0.9},      # 橡木
                'pine': {'density': 0.52, 'carbon_factor': 0.95},    # 松木
                'birch': {'density': 0.65, 'carbon_factor': 0.85},   # 樺木
                'maple': {'density': 0.70, 'carbon_factor': 0.88},   # 楓木
                'beech': {'density': 0.72, 'carbon_factor': 0.87},   # 山毛櫸
                'generic_wood': {'density': 0.65, 'carbon_factor': 0.90}  # 通用木材
            },
            'carbon_footprint': {
                'production_factor': 1.2,    # 生產階段碳足跡係數
                'transport_factor': 0.1,     # 運輸階段碳足跡係數
                'use_phase_factor': 0.05,    # 使用階段碳足跡係數
                'disposal_factor': 0.3,      # 廢棄階段碳足跡係數
            },
            'life_cycle_stages': {
                'material_extraction': 0.25,  # 原料開採階段佔比
                'production': 0.45,           # 生產製造階段佔比
                'transport': 0.10,            # 運輸配送階段佔比
                'use_phase': 0.15,            # 使用階段佔比
                'end_of_life': 0.05          # 報廢處理階段佔比
            }
        }
    
    def _initialize_material_database(self) -> Dict:
        """初始化材料數據庫"""
        return {
            'wood': {
                'name': '木材',
                'carbon_intensity': 0.9,  # kg CO2e/kg
                'recycling_rate': 0.85,
                'renewable': True,
                'lifespan': 25,  # 年
                'density_range': (0.4, 0.9),  # g/cm³
                'environmental_impact': {
                    'water_usage': 'low',
                    'land_use': 'medium',
                    'biodiversity': 'positive'
                }
            },
            'plastic': {
                'name': '塑膠',
                'carbon_intensity': 3.2,
                'recycling_rate': 0.25,
                'renewable': False,
                'lifespan': 15,
                'density_range': (0.9, 1.4),
                'environmental_impact': {
                    'water_usage': 'high',
                    'land_use': 'low',
                    'biodiversity': 'negative'
                }
            },
            'metal': {
                'name': '金屬',
                'carbon_intensity': 2.8,
                'recycling_rate': 0.95,
                'renewable': False,
                'lifespan': 50,
                'density_range': (2.7, 8.0),
                'environmental_impact': {
                    'water_usage': 'high',
                    'land_use': 'high',
                    'biodiversity': 'negative'
                }
            }
        }
    
    def analyze_chair_lca(self, chair_data: Dict) -> Dict:
        """
        分析單個椅子的LCA
        
        Args:
            chair_data: 椅子數據，包含幾何、重量、材料信息
            
        Returns:
            LCA分析結果
        """
        logger.info(f"開始LCA分析: {chair_data.get('chair_id', 'Unknown')}")
        
        # 1. 確定材料成分（預設為木頭）
        materials = self._determine_materials(chair_data)
        
        # 2. 計算體積和重量
        volume_mass = self._calculate_volume_and_mass(chair_data, materials)
        
        # 3. 生命週期各階段評估
        life_cycle_impact = self._calculate_life_cycle_impact(volume_mass, materials)
        
        # 4. 碳足跡計算
        carbon_footprint = self._calculate_carbon_footprint(volume_mass, materials)
        
        # 5. 環境影響評估
        environmental_impact = self._assess_environmental_impact(materials, volume_mass)
        
        # 6. 可持續性評分
        sustainability_score = self._calculate_sustainability_score(materials, life_cycle_impact)
        
        lca_result = {
            'chair_id': chair_data.get('chair_id', 'Unknown'),
            'analysis_timestamp': datetime.now().isoformat(),
            'materials': materials,
            'volume_mass': volume_mass,
            'life_cycle_impact': life_cycle_impact,
            'carbon_footprint': carbon_footprint,
            'environmental_impact': environmental_impact,
            'sustainability_score': sustainability_score,
            'recommendations': self._generate_recommendations(materials, sustainability_score)
        }
        
        self.analysis_results.append(lca_result)
        logger.info(f"LCA分析完成: {chair_data.get('chair_id', 'Unknown')}")
        
        return lca_result
    
    def _determine_materials(self, chair_data: Dict) -> Dict:
        """確定椅子材料成分（預設木頭）"""
        materials = {}
        
        # 檢查是否有材料檢測結果
        material_analysis = chair_data.get('material_analysis', {})
        
        if material_analysis and 'wood_percentage' in material_analysis:
            # 如果有材料分析結果，使用檢測到的木材比例
            wood_percentage = material_analysis['wood_percentage']
            if wood_percentage > 0:
                materials['wood'] = wood_percentage / 100.0
                materials['other'] = (100 - wood_percentage) / 100.0
            else:
                # 如果沒有檢測到木材，仍預設為木頭
                materials['wood'] = 0.9  # 90% 木材
                materials['other'] = 0.1  # 10% 其他材料
        else:
            # 沒有材料分析數據時，預設為木頭
            logger.info("未找到材料分析數據，預設材料為木頭")
            materials['wood'] = 0.95   # 95% 木材
            materials['metal'] = 0.05  # 5% 金屬（連接件）
        
        # 確定木材類型
        wood_type = chair_data.get('wood_type', 'generic_wood')
        if wood_type not in self.config['wood_types']:
            wood_type = 'generic_wood'
        
        materials['wood_type'] = wood_type
        materials['wood_properties'] = self.config['wood_types'][wood_type]
        
        return materials
    
    def _calculate_volume_and_mass(self, chair_data: Dict, materials: Dict) -> Dict:
        """計算體積和質量"""
        # 從幾何分析中獲取體積信息
        geometry_analysis = chair_data.get('geometry_analysis', {})
        
        # 估算體積（如果沒有直接數據）
        if 'bounding_box_volume' in geometry_analysis:
            bounding_volume = geometry_analysis['bounding_box_volume']
            # 椅子實際體積約為邊界框體積的30-50%
            estimated_volume = bounding_volume * 0.4  # cm³
        else:
            # 根據椅子尺寸估算
            seat_area = chair_data.get('seat_area', 400)  # cm²
            seat_thickness = chair_data.get('seat_thickness', 3)  # cm
            estimated_volume = seat_area * seat_thickness * 2.5  # 估算總體積
        
        # 計算質量
        wood_properties = materials.get('wood_properties', self.config['wood_types']['generic_wood'])
        wood_density = wood_properties['density']  # g/cm³
        
        # 質量計算
        wood_volume = estimated_volume * materials.get('wood', 0.95)
        metal_volume = estimated_volume * materials.get('metal', 0.05)
        
        wood_mass = wood_volume * wood_density  # g
        metal_mass = metal_volume * 2.7  # 假設鋁材密度 2.7 g/cm³
        
        total_mass = (wood_mass + metal_mass) / 1000  # 轉換為 kg
        
        return {
            'total_volume_cm3': estimated_volume,
            'wood_volume_cm3': wood_volume,
            'metal_volume_cm3': metal_volume,
            'total_mass_kg': total_mass,
            'wood_mass_kg': wood_mass / 1000,
            'metal_mass_kg': metal_mass / 1000,
            'density_g_cm3': total_mass * 1000 / estimated_volume
        }
    
    def _calculate_life_cycle_impact(self, volume_mass: Dict, materials: Dict) -> Dict:
        """計算生命週期各階段影響"""
        total_mass = volume_mass['total_mass_kg']
        wood_mass = volume_mass['wood_mass_kg']
        metal_mass = volume_mass['metal_mass_kg']
        
        stages = self.config['life_cycle_stages']
        
        # 各階段碳排放計算
        material_extraction = (
            wood_mass * self.material_database['wood']['carbon_intensity'] * stages['material_extraction'] +
            metal_mass * self.material_database['metal']['carbon_intensity'] * stages['material_extraction']
        )
        
        production = total_mass * self.config['carbon_footprint']['production_factor'] * stages['production']
        
        transport = total_mass * self.config['carbon_footprint']['transport_factor'] * stages['transport']
        
        use_phase = total_mass * self.config['carbon_footprint']['use_phase_factor'] * stages['use_phase']
        
        end_of_life = total_mass * self.config['carbon_footprint']['disposal_factor'] * stages['end_of_life']
        
        return {
            'material_extraction_kg_co2': material_extraction,
            'production_kg_co2': production,
            'transport_kg_co2': transport,
            'use_phase_kg_co2': use_phase,
            'end_of_life_kg_co2': end_of_life,
            'total_life_cycle_kg_co2': material_extraction + production + transport + use_phase + end_of_life
        }
    
    def _calculate_carbon_footprint(self, volume_mass: Dict, materials: Dict) -> Dict:
        """計算碳足跡"""
        wood_mass = volume_mass['wood_mass_kg']
        metal_mass = volume_mass['metal_mass_kg']
        
        # 材料碳足跡
        wood_carbon = wood_mass * self.material_database['wood']['carbon_intensity']
        metal_carbon = metal_mass * self.material_database['metal']['carbon_intensity']
        
        total_carbon = wood_carbon + metal_carbon
        
        # 碳效率評估
        carbon_efficiency = total_carbon / volume_mass['total_mass_kg']  # kg CO2e/kg
        
        return {
            'wood_carbon_kg_co2': wood_carbon,
            'metal_carbon_kg_co2': metal_carbon,
            'total_carbon_kg_co2': total_carbon,
            'carbon_efficiency_kg_co2_per_kg': carbon_efficiency,
            'carbon_grade': self._get_carbon_grade(carbon_efficiency)
        }
    
    def _assess_environmental_impact(self, materials: Dict, volume_mass: Dict) -> Dict:
        """評估環境影響"""
        wood_ratio = materials.get('wood', 0.95)
        
        # 基於木材比例的環境影響評分
        renewable_score = wood_ratio * 10  # 可再生性評分 (0-10)
        recycling_score = (
            wood_ratio * self.material_database['wood']['recycling_rate'] * 10 +
            (1 - wood_ratio) * self.material_database['metal']['recycling_rate'] * 10
        )
        
        # 水資源使用評分
        water_impact = wood_ratio * 2 + (1 - wood_ratio) * 8  # 木材用水較少
        
        # 土地使用評分
        land_impact = wood_ratio * 5 + (1 - wood_ratio) * 3
        
        # 總體環境評分
        overall_score = (renewable_score + recycling_score + (10 - water_impact) + land_impact) / 4
        
        return {
            'renewable_score': renewable_score,
            'recycling_score': recycling_score,
            'water_impact_score': 10 - water_impact,  # 翻轉分數，越低越好
            'land_impact_score': land_impact,
            'overall_environmental_score': overall_score,
            'environmental_grade': self._get_environmental_grade(overall_score)
        }
    
    def _calculate_sustainability_score(self, materials: Dict, life_cycle_impact: Dict) -> Dict:
        """計算可持續性評分"""
        wood_ratio = materials.get('wood', 0.95)
        total_carbon = life_cycle_impact['total_life_cycle_kg_co2']
        
        # 材料可持續性評分 (0-100)
        material_score = wood_ratio * 85 + (1 - wood_ratio) * 40
        
        # 碳足跡評分 (基於椅子的碳足跡，越低越好)
        carbon_score = max(0, 100 - total_carbon * 10)
        
        # 生命週期評分
        lifespan = self.material_database['wood']['lifespan'] * wood_ratio + \
                  self.material_database['metal']['lifespan'] * (1 - wood_ratio)
        lifespan_score = min(100, lifespan * 2)
        
        # 綜合評分
        overall_score = (material_score * 0.4 + carbon_score * 0.4 + lifespan_score * 0.2)
        
        return {
            'material_sustainability_score': material_score,
            'carbon_footprint_score': carbon_score,
            'lifespan_score': lifespan_score,
            'overall_sustainability_score': overall_score,
            'sustainability_grade': self._get_sustainability_grade(overall_score)
        }
    
    def _get_carbon_grade(self, carbon_efficiency: float) -> str:
        """獲取碳效率等級"""
        if carbon_efficiency <= 1.0:
            return 'A+'
        elif carbon_efficiency <= 1.5:
            return 'A'
        elif carbon_efficiency <= 2.0:
            return 'B'
        elif carbon_efficiency <= 2.5:
            return 'C'
        else:
            return 'D'
    
    def _get_environmental_grade(self, score: float) -> str:
        """獲取環境影響等級"""
        if score >= 8.5:
            return 'Excellent'
        elif score >= 7.0:
            return 'Good'
        elif score >= 5.5:
            return 'Fair'
        elif score >= 4.0:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _get_sustainability_grade(self, score: float) -> str:
        """獲取可持續性等級"""
        if score >= 85:
            return 'Highly Sustainable'
        elif score >= 70:
            return 'Sustainable'
        elif score >= 55:
            return 'Moderately Sustainable'
        elif score >= 40:
            return 'Low Sustainability'
        else:
            return 'Unsustainable'
    
    def _generate_recommendations(self, materials: Dict, sustainability_score: Dict) -> List[str]:
        """生成改進建議"""
        recommendations = []
        
        wood_ratio = materials.get('wood', 0.95)
        overall_score = sustainability_score['overall_sustainability_score']
        
        if wood_ratio < 0.8:
            recommendations.append("建議增加木材使用比例以提高可持續性")
        
        if overall_score < 70:
            recommendations.append("考慮使用認證可持續木材（如FSC認證）")
            recommendations.append("優化設計以減少材料使用量")
        
        if sustainability_score['carbon_footprint_score'] < 60:
            recommendations.append("考慮本地木材供應以減少運輸碳足跡")
            recommendations.append("優化生產工藝以降低能耗")
        
        if not recommendations:
            recommendations.append("當前設計已具有良好的環境表現")
            recommendations.append("繼續關注可持續材料的創新應用")
        
        return recommendations
    
    def analyze_batch_chairs(self, chairs_data: List[Dict]) -> Dict:
        """批量分析椅子LCA"""
        logger.info(f"開始批量LCA分析，共 {len(chairs_data)} 個椅子")
        
        batch_results = []
        summary_stats = {
            'total_chairs': len(chairs_data),
            'successful_analysis': 0,
            'failed_analysis': 0,
            'average_carbon_footprint': 0,
            'average_sustainability_score': 0,
            'material_distribution': {},
            'sustainability_grades': {}
        }
        
        for chair_data in chairs_data:
            try:
                result = self.analyze_chair_lca(chair_data)
                batch_results.append(result)
                summary_stats['successful_analysis'] += 1
            except Exception as e:
                logger.error(f"椅子LCA分析失敗: {chair_data.get('chair_id', 'Unknown')} - {e}")
                summary_stats['failed_analysis'] += 1
        
        # 計算統計摘要
        if batch_results:
            carbon_footprints = [r['carbon_footprint']['total_carbon_kg_co2'] for r in batch_results]
            sustainability_scores = [r['sustainability_score']['overall_sustainability_score'] for r in batch_results]
            
            summary_stats['average_carbon_footprint'] = np.mean(carbon_footprints)
            summary_stats['average_sustainability_score'] = np.mean(sustainability_scores)
            
            # 材料分布統計
            for result in batch_results:
                wood_ratio = result['materials'].get('wood', 0)
                if wood_ratio >= 0.9:
                    category = 'High Wood Content (≥90%)'
                elif wood_ratio >= 0.7:
                    category = 'Medium Wood Content (70-89%)'
                else:
                    category = 'Low Wood Content (<70%)'
                
                summary_stats['material_distribution'][category] = \
                    summary_stats['material_distribution'].get(category, 0) + 1
            
            # 可持續性等級分布
            for result in batch_results:
                grade = result['sustainability_score']['sustainability_grade']
                summary_stats['sustainability_grades'][grade] = \
                    summary_stats['sustainability_grades'].get(grade, 0) + 1
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'summary_statistics': summary_stats,
            'individual_results': batch_results,
            'recommendations': self._generate_batch_recommendations(summary_stats)
        }
    
    def _generate_batch_recommendations(self, summary_stats: Dict) -> List[str]:
        """生成批量分析建議"""
        recommendations = []
        
        avg_carbon = summary_stats['average_carbon_footprint']
        avg_sustainability = summary_stats['average_sustainability_score']
        
        if avg_carbon > 3.0:
            recommendations.append("整體碳足跡偏高，建議優化材料選擇和生產工藝")
        
        if avg_sustainability < 70:
            recommendations.append("整體可持續性有提升空間，建議增加可持續材料使用")
        
        # 基於材料分布的建議
        material_dist = summary_stats['material_distribution']
        if material_dist.get('Low Wood Content (<70%)', 0) > 0:
            recommendations.append("部分椅子木材含量較低，建議提高木材使用比例")
        
        return recommendations
    
    def save_lca_report(self, results: Dict, output_dir: str) -> str:
        """保存LCA分析報告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON報告
        json_file = output_path / f'lca_analysis_report_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 文字摘要報告
        txt_file = output_path / f'lca_analysis_summary_{timestamp}.txt'
        with open(txt_file, 'w', encoding='utf-8') as f:
            self._write_summary_report(f, results)
        
        logger.info(f"LCA分析報告已保存: {json_file}")
        logger.info(f"LCA摘要報告已保存: {txt_file}")
        
        return str(json_file)
    
    def _write_summary_report(self, f, results: Dict):
        """寫入摘要報告"""
        f.write("=" * 80 + "\n")
        f.write("椅子生命週期評估 (LCA) 分析報告\n")
        f.write("=" * 80 + "\n\n")
        
        summary = results['summary_statistics']
        
        f.write(f"分析時間: {results['analysis_timestamp']}\n")
        f.write(f"分析椅子總數: {summary['total_chairs']}\n")
        f.write(f"成功分析: {summary['successful_analysis']}\n")
        f.write(f"分析失敗: {summary['failed_analysis']}\n\n")
        
        if summary['successful_analysis'] > 0:
            f.write("整體統計結果:\n")
            f.write("-" * 40 + "\n")
            f.write(f"平均碳足跡: {summary['average_carbon_footprint']:.2f} kg CO2e\n")
            f.write(f"平均可持續性評分: {summary['average_sustainability_score']:.1f}/100\n\n")
            
            f.write("材料分布:\n")
            f.write("-" * 40 + "\n")
            for category, count in summary['material_distribution'].items():
                percentage = count / summary['successful_analysis'] * 100
                f.write(f"{category}: {count} 個 ({percentage:.1f}%)\n")
            
            f.write("\n可持續性等級分布:\n")
            f.write("-" * 40 + "\n")
            for grade, count in summary['sustainability_grades'].items():
                percentage = count / summary['successful_analysis'] * 100
                f.write(f"{grade}: {count} 個 ({percentage:.1f}%)\n")
            
            f.write("\n改進建議:\n")
            f.write("-" * 40 + "\n")
            for i, rec in enumerate(results['recommendations'], 1):
                f.write(f"{i}. {rec}\n")


def integrate_with_workflow(chairs_data: List[Dict], config: Dict = None) -> Dict:
    """
    整合到主工作流程的LCA分析函數
    
    Args:
        chairs_data: 椅子數據列表
        config: LCA配置參數
        
    Returns:
        LCA分析結果
    """
    try:
        analyzer = EnhancedLCAAnalyzer(config)
        
        if not chairs_data:
            logger.warning("沒有椅子數據可進行LCA分析")
            return {
                'success': False,
                'error': 'No chair data available for LCA analysis',
                'message': '沒有足夠的數據進行LCA分析'
            }
        
        logger.info(f"開始LCA分析，共 {len(chairs_data)} 個椅子")
        results = analyzer.analyze_batch_chairs(chairs_data)
        
        return {
            'success': True,
            'results': results,
            'message': f'成功分析 {results["summary_statistics"]["successful_analysis"]} 個椅子的LCA'
        }
        
    except Exception as e:
        logger.error(f"LCA分析異常: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': '❌ LCA分析階段失敗'
        }


def main():
    """測試函數"""
    # 模擬椅子數據
    test_chairs = [
        {
            'chair_id': 'Chair_002',
            'seat_area': 350,
            'seat_thickness': 2.5,
            'geometry_analysis': {
                'bounding_box_volume': 2500
            },
            'material_analysis': {
                'wood_percentage': 92
            }
        }
    ]
    
    # 執行LCA分析
    analyzer = EnhancedLCAAnalyzer()
    results = analyzer.analyze_batch_chairs(test_chairs)
    
    # 保存報告
    output_dir = './lca_reports'
    report_file = analyzer.save_lca_report(results, output_dir)
    
    print(f"LCA分析完成，報告保存至: {report_file}")
    print(f"平均碳足跡: {results['summary_statistics']['average_carbon_footprint']:.2f} kg CO2e")
    print(f"平均可持續性評分: {results['summary_statistics']['average_sustainability_score']:.1f}/100")
