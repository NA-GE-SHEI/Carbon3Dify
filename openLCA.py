import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns


# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MaterialProperties:
    """材料屬性數據類"""
    name: str
    density: float  # kg/m³
    carbon_factor: float  # kg CO2/kg material
    price_per_kg: float  # $/kg
    recyclability: float  # 0-1 recyclability factor
    durability_years: int  # expected lifespan
    processing_energy: float  # MJ/kg processing energy


@dataclass
class ChairGeometry:
    """椅子幾何數據類"""
    total_volume: float  # m³
    seat_area: float  # m²
    seat_thickness: float  # m
    back_height: float  # m
    back_volume: float  # m³
    leg_volume: float  # m³
    seat_volume: float  # m³
    predicted_weight: float  # kg
    surface_area: float  # m²


@dataclass
class LCAResult:
    """LCA計算結果數據類"""
    total_carbon_footprint: float  # kg CO2 equivalent
    material_carbon: float  # kg CO2 from materials
    manufacturing_carbon: float  # kg CO2 from manufacturing
    transport_carbon: float  # kg CO2 from transportation
    use_phase_carbon: float  # kg CO2 from use phase
    end_of_life_carbon: float  # kg CO2 from disposal/recycling
    total_cost: float  # total cost
    environmental_score: float  # 0-100 environmental performance score


class OpenLCACalculator:
    """openLCA生命週期評估計算器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化LCA計算器
        
        Args:
            config_file: 配置文件路徑
        """
        self.config = self.load_config(config_file)
        self.materials_db = self.load_materials_database()
        self.lca_results = {}
        
    def load_config(self, config_file: Optional[str] = None) -> Dict:
        """載入配置文件"""
        default_config = {
            "materials": {
                "default_material": "wood_pine",
                "material_distribution": {
                    "seat": {"wood_pine": 0.8, "foam": 0.2},
                    "back": {"wood_pine": 0.9, "fabric": 0.1},
                    "legs": {"wood_pine": 1.0}
                }
            },
            "manufacturing": {
                "energy_intensity": 2.5,  # MJ/kg
                "carbon_factor": 0.5,  # kg CO2/MJ
                "waste_factor": 0.05,  # 5% material waste
            },
            "transportation": {
                "distance_km": 500,  # average transport distance
                "mode": "truck",  # truck, ship, air
                "carbon_factor": 0.12  # kg CO2/kg/km for truck
            },
            "use_phase": {
                "lifespan_years": 15,
                "maintenance_carbon_per_year": 0.1  # kg CO2/year
            },
            "end_of_life": {
                "recycling_rate": 0.6,
                "recycling_benefit": -0.5,  # kg CO2/kg (negative = benefit)
                "disposal_carbon": 0.1  # kg CO2/kg
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    self._deep_update(default_config, user_config)
                logger.info(f"✅ 已載入LCA配置文件: {config_file}")
            except Exception as e:
                logger.warning(f"⚠️  載入配置文件失敗，使用默認配置: {e}")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def load_materials_database(self) -> Dict[str, MaterialProperties]:
        """載入材料數據庫"""
        materials = {
            "wood_pine": MaterialProperties(
                name="Pine Wood",
                density=500,  # kg/m³
                carbon_factor=0.45,  # kg CO2/kg (including forestry)
                price_per_kg=2.50,
                recyclability=0.9,
                durability_years=15,
                processing_energy=3.5
            ),
            "wood_oak": MaterialProperties(
                name="Oak Wood",
                density=700,
                carbon_factor=0.52,
                price_per_kg=8.00,
                recyclability=0.95,
                durability_years=25,
                processing_energy=4.2
            ),
            "plastic_pp": MaterialProperties(
                name="Polypropylene",
                density=900,
                carbon_factor=2.1,
                price_per_kg=1.80,
                recyclability=0.7,
                durability_years=20,
                processing_energy=78.0
            ),
            "metal_steel": MaterialProperties(
                name="Steel",
                density=7850,
                carbon_factor=2.3,
                price_per_kg=1.20,
                recyclability=0.95,
                durability_years=30,
                processing_energy=25.0
            ),
            "foam": MaterialProperties(
                name="Polyurethane Foam",
                density=30,
                carbon_factor=3.8,
                price_per_kg=3.50,
                recyclability=0.2,
                durability_years=10,
                processing_energy=95.0
            ),
            "fabric": MaterialProperties(
                name="Cotton Fabric",
                density=200,
                carbon_factor=5.2,
                price_per_kg=12.00,
                recyclability=0.8,
                durability_years=8,
                processing_energy=55.0
            )
        }
        
        logger.info(f"✅ 載入了 {len(materials)} 種材料的數據")
        return materials
    
    def extract_geometry_from_analysis(self, analysis_results_dir: str) -> Optional[ChairGeometry]:
        """
        從分析結果中提取幾何參數
        
        Args:
            analysis_results_dir: 分析結果目錄
            
        Returns:
            ChairGeometry對象或None
        """
        try:
            # 從enhanced_analysis結果中讀取預測重量
            enhanced_dir = Path(analysis_results_dir) / "enhanced_analysis"
            predicted_weight = 4.2  # 默認值
            
            # 查找results.txt文件
            results_files = list(enhanced_dir.rglob("results.txt"))
            if results_files:
                with open(results_files[0], 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 提取預測重量
                    import re
                    weight_match = re.search(r"預測重量:\s*([\d\.]+)", content)
                    if weight_match:
                        predicted_weight = float(weight_match.group(1))
            
            # 從配置或分析結果中獲取幾何參數
            geometry = ChairGeometry(
                total_volume=0.003,  # 默認3升
                seat_area=0.0707,  # 707 cm² = 0.0707 m²
                seat_thickness=0.03,  # 3cm
                back_height=0.40,  # 40cm
                back_volume=0.00039,  # 390 cm³
                leg_volume=0.00155,  # 1550 cm³
                seat_volume=0.002771,  # 2771 cm³
                predicted_weight=predicted_weight,
                surface_area=0.5  # 估計表面積
            )
            
            logger.info(f"✅ 提取幾何參數: 預測重量 {predicted_weight:.2f} kg")
            return geometry
            
        except Exception as e:
            logger.error(f"❌ 提取幾何參數失敗: {e}")
            return None
    
    def calculate_material_carbon(self, geometry: ChairGeometry, material_type: str = "wood_pine") -> Tuple[float, Dict]:
        """
        計算材料階段碳足跡
        
        Args:
            geometry: 椅子幾何參數
            material_type: 主要材料類型
            
        Returns:
            碳足跡值和詳細分解
        """
        if material_type not in self.materials_db:
            material_type = "wood_pine"  # 默認材料
        
        material = self.materials_db[material_type]
        
        # 計算各部件材料用量
        material_breakdown = {}
        
        # 座椅部分
        seat_weight = geometry.seat_volume * material.density
        material_breakdown["seat"] = {
            "weight_kg": seat_weight,
            "carbon_kg": seat_weight * material.carbon_factor
        }
        
        # 椅背部分
        back_weight = geometry.back_volume * material.density
        material_breakdown["back"] = {
            "weight_kg": back_weight,
            "carbon_kg": back_weight * material.carbon_factor
        }
        
        # 椅腳部分
        legs_weight = geometry.leg_volume * material.density
        material_breakdown["legs"] = {
            "weight_kg": legs_weight,
            "carbon_kg": legs_weight * material.carbon_factor
        }
        
        # 添加材料浪費
        waste_factor = self.config["manufacturing"]["waste_factor"]
        total_material_weight = seat_weight + back_weight + legs_weight
        waste_weight = total_material_weight * waste_factor
        waste_carbon = waste_weight * material.carbon_factor
        
        material_breakdown["waste"] = {
            "weight_kg": waste_weight,
            "carbon_kg": waste_carbon
        }
        
        # 總材料碳足跡
        total_carbon = sum(item["carbon_kg"] for item in material_breakdown.values())
        
        logger.info(f"✅ 材料碳足跡: {total_carbon:.3f} kg CO2")
        return total_carbon, material_breakdown
    
    def calculate_manufacturing_carbon(self, geometry: ChairGeometry) -> float:
        """計算製造階段碳足跡"""
        config = self.config["manufacturing"]
        
        # 基於重量的製造能耗
        energy_consumption = geometry.predicted_weight * config["energy_intensity"]  # MJ
        manufacturing_carbon = energy_consumption * config["carbon_factor"]  # kg CO2
        
        logger.info(f"✅ 製造碳足跡: {manufacturing_carbon:.3f} kg CO2")
        return manufacturing_carbon
    
    def calculate_transportation_carbon(self, geometry: ChairGeometry) -> float:
        """計算運輸階段碳足跡"""
        config = self.config["transportation"]
        
        # 運輸碳足跡 = 重量 × 距離 × 碳排放因子
        transport_carbon = (geometry.predicted_weight * 
                          config["distance_km"] * 
                          config["carbon_factor"])
        
        logger.info(f"✅ 運輸碳足跡: {transport_carbon:.3f} kg CO2")
        return transport_carbon
    
    def calculate_use_phase_carbon(self, geometry: ChairGeometry) -> float:
        """計算使用階段碳足跡"""
        config = self.config["use_phase"]
        
        # 使用階段主要是維護產生的碳排放
        use_carbon = config["lifespan_years"] * config["maintenance_carbon_per_year"]
        
        logger.info(f"✅ 使用階段碳足跡: {use_carbon:.3f} kg CO2")
        return use_carbon
    
    def calculate_end_of_life_carbon(self, geometry: ChairGeometry, material_type: str = "wood_pine") -> float:
        """計算生命周期結束階段碳足跡"""
        config = self.config["end_of_life"]
        weight = geometry.predicted_weight
        
        # 回收部分的環境效益（負值）
        recycling_weight = weight * config["recycling_rate"]
        recycling_benefit = recycling_weight * config["recycling_benefit"]
        
        # 剩餘部分的處置碳足跡
        disposal_weight = weight * (1 - config["recycling_rate"])
        disposal_carbon = disposal_weight * config["disposal_carbon"]
        
        total_eol_carbon = recycling_benefit + disposal_carbon
        
        logger.info(f"✅ 生命週期結束碳足跡: {total_eol_carbon:.3f} kg CO2")
        return total_eol_carbon
    
    def calculate_total_cost(self, geometry: ChairGeometry, material_type: str = "wood_pine") -> float:
        """計算總成本"""
        if material_type not in self.materials_db:
            material_type = "wood_pine"
        
        material = self.materials_db[material_type]
        
        # 材料成本
        material_cost = geometry.predicted_weight * material.price_per_kg
        
        # 製造成本（假設為材料成本的2倍）
        manufacturing_cost = material_cost * 2
        
        # 運輸成本（假設為材料成本的10%）
        transport_cost = material_cost * 0.1
        
        total_cost = material_cost + manufacturing_cost + transport_cost
        
        logger.info(f"✅ 總成本: ${total_cost:.2f}")
        return total_cost
    
    def calculate_environmental_score(self, lca_result: LCAResult) -> float:
        """
        計算環境性能評分 (0-100)
        分數越高表示環境性能越好
        """
        # 基準值（假設的行業平均值）
        baseline_carbon = 15.0  # kg CO2
        baseline_cost = 100.0  # $
        
        # 碳足跡評分（50%權重）
        carbon_score = max(0, 50 * (1 - lca_result.total_carbon_footprint / baseline_carbon))
        
        # 成本效益評分（30%權重）
        cost_score = max(0, 30 * (1 - lca_result.total_cost / baseline_cost))
        
        # 可回收性評分（20%權重）- 基於材料的可回收性
        recyclability_score = 20 * 0.9  # 假設木材的可回收性為90%
        
        total_score = carbon_score + cost_score + recyclability_score
        
        logger.info(f"✅ 環境評分: {total_score:.1f}/100")
        return min(100, max(0, total_score))
    
    def perform_lca_analysis(self, analysis_results_dir: str, material_type: str = "wood_pine") -> Optional[LCAResult]:
        """
        執行完整的LCA分析
        
        Args:
            analysis_results_dir: 分析結果目錄
            material_type: 材料類型
            
        Returns:
            LCA結果對象
        """
        logger.info("🔄 開始生命週期評估(LCA)分析")
        logger.info("=" * 60)
        
        # 提取幾何參數
        geometry = self.extract_geometry_from_analysis(analysis_results_dir)
        if not geometry:
            logger.error("❌ 無法提取幾何參數，LCA分析終止")
            return None
        
        # 計算各階段碳足跡
        material_carbon, material_breakdown = self.calculate_material_carbon(geometry, material_type)
        manufacturing_carbon = self.calculate_manufacturing_carbon(geometry)
        transport_carbon = self.calculate_transportation_carbon(geometry)
        use_phase_carbon = self.calculate_use_phase_carbon(geometry)
        end_of_life_carbon = self.calculate_end_of_life_carbon(geometry, material_type)
        
        # 計算總碳足跡
        total_carbon = (material_carbon + manufacturing_carbon + transport_carbon + 
                       use_phase_carbon + end_of_life_carbon)
        
        # 計算總成本
        total_cost = self.calculate_total_cost(geometry, material_type)
        
        # 創建LCA結果對象
        lca_result = LCAResult(
            total_carbon_footprint=total_carbon,
            material_carbon=material_carbon,
            manufacturing_carbon=manufacturing_carbon,
            transport_carbon=transport_carbon,
            use_phase_carbon=use_phase_carbon,
            end_of_life_carbon=end_of_life_carbon,
            total_cost=total_cost,
            environmental_score=0  # 臨時值，稍後計算
        )
        
        # 計算環境評分
        lca_result.environmental_score = self.calculate_environmental_score(lca_result)
        
        # 保存結果
        self.lca_results[material_type] = {
            'geometry': geometry,
            'lca_result': lca_result,
            'material_breakdown': material_breakdown,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("=" * 60)
        logger.info("🎉 LCA分析完成！")
        logger.info(f"總碳足跡: {total_carbon:.3f} kg CO2")
        logger.info(f"環境評分: {lca_result.environmental_score:.1f}/100")
        
        return lca_result
    
    def save_lca_results(self, output_dir: str = "./lca_results"):
        """保存LCA分析結果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON格式詳細結果
        json_file = output_path / f'lca_results_{timestamp}.json'
        
        # 準備可序列化的數據
        serializable_data = {}
        for material_type, data in self.lca_results.items():
            serializable_data[material_type] = {
                'geometry': asdict(data['geometry']),
                'lca_result': asdict(data['lca_result']),
                'material_breakdown': data['material_breakdown'],
                'timestamp': data['timestamp']
            }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        # 保存文字格式摘要報告
        text_file = output_path / f'lca_summary_{timestamp}.txt'
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("椅子產品生命週期評估(LCA)報告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for material_type, data in self.lca_results.items():
                geometry = data['geometry']
                lca_result = data['lca_result']
                
                f.write(f"材料類型: {material_type}\n")
                f.write("-" * 30 + "\n")
                f.write(f"預測重量: {geometry.predicted_weight:.2f} kg\n")
                f.write(f"總體積: {geometry.total_volume*1000:.1f} L\n\n")
                
                f.write("碳足跡分解 (kg CO2):\n")
                f.write(f"  材料階段: {lca_result.material_carbon:.3f}\n")
                f.write(f"  製造階段: {lca_result.manufacturing_carbon:.3f}\n")
                f.write(f"  運輸階段: {lca_result.transport_carbon:.3f}\n")
                f.write(f"  使用階段: {lca_result.use_phase_carbon:.3f}\n")
                f.write(f"  生命週期結束: {lca_result.end_of_life_carbon:.3f}\n")
                f.write(f"  總計: {lca_result.total_carbon_footprint:.3f}\n\n")
                
                f.write(f"總成本: ${lca_result.total_cost:.2f}\n")
                f.write(f"環境評分: {lca_result.environmental_score:.1f}/100\n")
                f.write("\n" + "="*50 + "\n\n")
        
        logger.info(f"✅ LCA結果已保存:")
        logger.info(f"  - 詳細結果: {json_file}")
        logger.info(f"  - 摘要報告: {text_file}")
    
    def create_lca_visualization(self, output_dir: str = "./lca_results"):
        """創建LCA結果可視化圖表"""
        if not self.lca_results:
            logger.warning("⚠️  沒有LCA結果可用於可視化")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 設置字體（移除中文字體設置）
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 創建圖表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        for material_type, data in self.lca_results.items():
            lca_result = data['lca_result']
            
            # 1. 碳足跡分解餅圖 - 改為英文標籤
            labels = ['Material', 'Manufacturing', 'Transport', 'Use Phase', 'End of Life']
            sizes = [
                lca_result.material_carbon,
                lca_result.manufacturing_carbon,
                lca_result.transport_carbon,
                lca_result.use_phase_carbon,
                abs(lca_result.end_of_life_carbon)  # 取絕對值用於顯示
            ]
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'{material_type} - Carbon Footprint Breakdown')
            
            # 2. 各階段碳足跡柱狀圖 - 改為英文標籤
            phases = ['Material', 'Manufacturing', 'Transport', 'Use Phase', 'End of\nLife']
            carbon_values = [
                lca_result.material_carbon,
                lca_result.manufacturing_carbon,
                lca_result.transport_carbon,
                lca_result.use_phase_carbon,
                lca_result.end_of_life_carbon
            ]
            
            bars = ax2.bar(phases, carbon_values, color=colors)
            ax2.set_title(f'{material_type} - Carbon Footprint by Phase')
            ax2.set_ylabel('Carbon Emissions (kg CO2)')
            ax2.tick_params(axis='x', rotation=45)
            
            # 在柱子上添加數值標籤
            for bar, value in zip(bars, carbon_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height > 0 else height - 0.2,
                        f'{value:.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 3. 環境性能雷達圖 - 改為英文標籤
        categories = ['Carbon Footprint\n(Low=Good)', 'Cost Efficiency', 'Recyclability', 'Durability', 'Maintainability']
        
        # 為每種材料創建雷達圖數據
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 閉合圖形
        
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        
        for i, (material_type, data) in enumerate(self.lca_results.items()):
            lca_result = data['lca_result']
            material = self.materials_db.get(material_type, self.materials_db['wood_pine'])
            
            # 計算各維度評分 (0-10)
            carbon_score = max(0, 10 - lca_result.total_carbon_footprint / 2)  # 碳足跡越低分數越高
            cost_score = max(0, 10 - lca_result.total_cost / 20)  # 成本越低分數越高
            recyclability_score = material.recyclability * 10
            durability_score = min(10, material.durability_years / 3)
            maintainability_score = 8.0  # 假設值
            
            values = [carbon_score, cost_score, recyclability_score, durability_score, maintainability_score]
            values += values[:1]  # 閉合圖形
            
            ax3.plot(angles, values, 'o-', linewidth=2, label=material_type)
            ax3.fill(angles, values, alpha=0.25)
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 10)
        ax3.set_title('Environmental Performance Radar Chart')
        ax3.legend()
        
        # 4. 成本vs碳足跡散點圖 - 改為英文標籤
        materials = list(self.lca_results.keys())
        costs = [self.lca_results[mat]['lca_result'].total_cost for mat in materials]
        carbons = [self.lca_results[mat]['lca_result'].total_carbon_footprint for mat in materials]
        
        ax4.scatter(costs, carbons, s=100, alpha=0.7)
        
        for i, mat in enumerate(materials):
            ax4.annotate(mat, (costs[i], carbons[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
        
        ax4.set_xlabel('Total Cost ($)')
        ax4.set_ylabel('Total Carbon Footprint (kg CO2)')
        ax4.set_title('Cost vs Carbon Footprint')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = output_path / f'lca_visualization_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ LCA可視化圖表已保存: {plot_file}")
    
    def compare_materials(self, analysis_results_dir: str, materials: List[str] = None) -> Dict:
        """
        比較不同材料的LCA結果
        
        Args:
            analysis_results_dir: 分析結果目錄
            materials: 要比較的材料列表
            
        Returns:
            比較結果字典
        """
        if materials is None:
            materials = ['wood_pine', 'wood_oak', 'plastic_pp', 'metal_steel']
        
        logger.info(f"🔄 開始材料比較分析: {materials}")
        
        comparison_results = {}
        
        for material in materials:
            if material in self.materials_db:
                lca_result = self.perform_lca_analysis(analysis_results_dir, material)
                if lca_result:
                    comparison_results[material] = lca_result
        
        # 創建比較表
        if comparison_results:
            self.create_comparison_table(comparison_results)
        
        return comparison_results
    
    def create_comparison_table(self, comparison_results: Dict[str, LCAResult]):
        """創建材料比較表"""
        data = []
        
        for material, lca_result in comparison_results.items():
            material_props = self.materials_db[material]
            data.append({
                '材料': material_props.name,
                '總碳足跡 (kg CO2)': f"{lca_result.total_carbon_footprint:.3f}",
                '材料碳足跡': f"{lca_result.material_carbon:.3f}",
                '製造碳足跡': f"{lca_result.manufacturing_carbon:.3f}",
                '總成本 ($)': f"{lca_result.total_cost:.2f}",
                '環境評分': f"{lca_result.environmental_score:.1f}/100",
                '可回收性': f"{material_props.recyclability:.1%}",
                '預期壽命': f"{material_props.durability_years}年"
            })
        
        df = pd.DataFrame(data)
        
        # 保存比較表
        output_dir = Path("./lca_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = output_dir / f'material_comparison_{timestamp}.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        logger.info("📊 材料比較表:")
        logger.info("\n" + df.to_string(index=False))
        logger.info(f"\n✅ 比較表已保存: {csv_file}")



def main():
    """主函數 - 示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='椅子產品生命週期評估(LCA)計算器')
    parser.add_argument('--analysis_dir', type=str, default='.',
                       help='工作流程分析結果目錄')
    parser.add_argument('--material', type=str, default='wood_pine',
                       choices=['wood_pine', 'wood_oak', 'plastic_pp', 'metal_steel'],
                       help='材料類型')
    parser.add_argument('--compare', action='store_true',
                       help='比較不同材料')
    parser.add_argument('--config', type=str, default=None,
                       help='LCA配置文件路徑')
    parser.add_argument('--output_dir', type=str, default='./lca_results',
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    # 初始化LCA計算器
    calculator = OpenLCACalculator(args.config)
    
    try:
        if args.compare:
            # 材料比較模式
            logger.info("🔄 執行材料比較分析...")
            comparison_results = calculator.compare_materials(args.analysis_dir)
            
            if comparison_results:
                logger.info(f"✅ 成功比較了 {len(comparison_results)} 種材料")
            else:
                logger.error("❌ 材料比較失敗")
        else:
            # 單一材料分析模式
            logger.info(f"🔄 執行 {args.material} 材料的LCA分析...")
            lca_result = calculator.perform_lca_analysis(args.analysis_dir, args.material)
            
            if lca_result:
                logger.info("✅ LCA分析完成")
                logger.info(f"總碳足跡: {lca_result.total_carbon_footprint:.3f} kg CO2")
                logger.info(f"總成本: ${lca_result.total_cost:.2f}")
                logger.info(f"環境評分: {lca_result.environmental_score:.1f}/100")
            else:
                logger.error("❌ LCA分析失敗")
        
        # 保存結果和創建可視化
        calculator.save_lca_results(args.output_dir)
        calculator.create_lca_visualization(args.output_dir)
        
        logger.info("🎉 所有LCA分析完成！")
        
    except Exception as e:
        logger.error(f"❌ LCA分析過程中發生錯誤: {e}")
        return 1
    
    return 0



if __name__ == "__main__":
    exit(main())
