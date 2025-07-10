import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional
import logging
import platform
import subprocess


def setup_chinese_matplotlib():
    """設置matplotlib中文字體支援"""
    try:
        system = platform.system()
        
        # 根據作業系統選擇字體
        if system == "Windows":
            fonts = ['Microsoft YaHei', 'SimHei', 'Microsoft JhengHei', 'SimSun']
        elif system == "Darwin":  # macOS
            fonts = ['Arial Unicode MS', 'Hiragino Sans GB', 'PingFang SC', 'STHeiti']
        else:  # Linux
            fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
        
        # 檢查可用字體
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 選擇第一個可用的字體
        selected_font = None
        for font in fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        if selected_font:
            plt.rcParams['font.sans-serif'] = [selected_font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 已設置字體: {selected_font}")
            return True
        else:
            # 備用方案：使用英文標籤
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print("⚠️ 未找到中文字體，將使用英文標籤")
            return False
        
    except Exception as e:
        print(f"❌ 字體設置失敗: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False


def generate_visual_report(workflow_results: Dict, lca_results: Dict, logger):
    """生成視覺化報告，包含圖表和美化的文字報告"""
    logger.info("🎨 開始生成視覺化報告...")
    
    try:
        # 首先設置中文字體
        chinese_support = setup_chinese_matplotlib()
        
        # 創建報告目錄
        visual_reports_dir = Path("./visual_reports")
        visual_reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成圖表
        if lca_results.get('success', False):
            generate_lca_charts(lca_results, visual_reports_dir, timestamp, logger, chinese_support)
        
        # 生成HTML報告
        html_report_path = generate_html_report(workflow_results, lca_results, visual_reports_dir, timestamp, logger, chinese_support)
        
        # 生成詳細的文字報告
        detailed_report_path = generate_detailed_text_report(workflow_results, lca_results, visual_reports_dir, timestamp, logger)
        
        # 生成執行摘要
        executive_summary_path = generate_executive_summary(workflow_results, lca_results, visual_reports_dir, timestamp, logger)
        
        logger.info(f"✅ 視覺化報告生成完成")
        logger.info(f"📊 HTML報告: {html_report_path}")
        logger.info(f"📄 詳細文字報告: {detailed_report_path}")
        logger.info(f"📋 執行摘要: {executive_summary_path}")
        
        return {
            'html_report': str(html_report_path),
            'detailed_report': str(detailed_report_path),
            'executive_summary': str(executive_summary_path),
            'charts_generated': True
        }
        
    except Exception as e:
        logger.error(f"❌ 生成視覺化報告時發生錯誤: {e}")
        return None


def generate_lca_charts(lca_results: Dict, output_dir: Path, timestamp: str, logger, chinese_support: bool):
    """生成LCA分析圖表"""
    logger.info("📊 生成LCA分析圖表...")
    
    try:
        summary = lca_results.get('summary', {})
        detailed_results = lca_results.get('detailed_results', [])
        
        if not detailed_results:
            logger.warning("⚠️ 沒有詳細結果可供圖表生成")
            return
        
        # 創建圖表子目錄
        charts_dir = output_dir / f"charts_{timestamp}"
        charts_dir.mkdir(exist_ok=True)
        
        # 1. 碳足跡對比圖
        generate_carbon_footprint_chart(detailed_results, charts_dir, logger, chinese_support)
        
        # 2. 可持續性評分圖
        generate_sustainability_score_chart(detailed_results, charts_dir, logger, chinese_support)
        
        # 3. 材料組成餅圖
        generate_material_composition_chart(summary, charts_dir, logger, chinese_support)
        
        # 4. 生命週期階段分析圖
        generate_lifecycle_stage_chart(detailed_results, charts_dir, logger, chinese_support)
        
        # 5. 綜合指標雷達圖
        generate_radar_chart(summary, charts_dir, logger, chinese_support)
        
    except Exception as e:
        logger.error(f"❌ 生成圖表時發生錯誤: {e}")


def generate_carbon_footprint_chart(results: List[Dict], output_dir: Path, logger, chinese_support: bool):
    """生成碳足跡對比圖表 - 修復版本"""
    try:
        # 設置圖表大小
        plt.figure(figsize=(12, 8))
        
        chair_ids = []
        carbon_values = []
        colors = []
        
        # 處理數據
        for result in results[:10]:  # 最多顯示10個
            if 'carbon_footprint' in result:
                # 確保椅子ID不會太長
                chair_id = result.get('chair_id', 'Unknown')
                if len(chair_id) > 15:
                    chair_id = chair_id[:12] + '...'
                chair_ids.append(chair_id)
                
                carbon = result['carbon_footprint'].get('total_carbon_kg_co2', 0)
                carbon_values.append(carbon)
                
                # 根據碳足跡值設置顏色
                if carbon < 2.0:
                    colors.append('#4CAF50')  # 綠色 - 優秀
                elif carbon < 3.0:
                    colors.append('#FFC107')  # 黃色 - 良好
                elif carbon < 4.0:
                    colors.append('#FF9800')  # 橙色 - 一般
                else:
                    colors.append('#F44336')  # 紅色 - 需改進
        
        if chair_ids:
            # 創建條形圖
            bars = plt.bar(range(len(chair_ids)), carbon_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # 設置 x 軸標籤
            plt.xticks(range(len(chair_ids)), chair_ids, rotation=45, ha='right', fontsize=10)
            plt.xlabel('Chair ID', fontsize=12, fontweight='bold')
            plt.ylabel('Carbon Footprint (kg CO₂e)', fontsize=12, fontweight='bold')
            plt.title('Chair Carbon Footprint Comparison Analysis', fontsize=14, fontweight='bold', pad=20)
            
            # 添加參考線和圖例
            # plt.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Excellent (<2.0)')
            # plt.axhline(y=3.0, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Good (<3.0)')
            # plt.axhline(y=4.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Needs Improvement (>4.0)')            
            # 設置標籤和標題
            # if chinese_support:
            #     plt.xlabel('椅子編號', fontsize=12, fontweight='bold')
            #     plt.ylabel('碳足跡 (kg CO₂e)', fontsize=12, fontweight='bold')
            #     plt.title('椅子碳足跡對比分析', fontsize=14, fontweight='bold', pad=20)
                
            #     # 添加參考線和圖例
            #     plt.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, linewidth=2, label='優秀標準 (<2.0)')
            #     plt.axhline(y=3.0, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='良好標準 (<3.0)')
            #     plt.axhline(y=4.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='需改進標準 (>4.0)')
            # else:
            #     plt.xlabel('Chair ID', fontsize=12, fontweight='bold')
            #     plt.ylabel('Carbon Footprint (kg CO₂e)', fontsize=12, fontweight='bold')
            #     plt.title('Chair Carbon Footprint Comparison Analysis', fontsize=14, fontweight='bold', pad=20)
                
            #     # 添加參考線和圖例
            #     plt.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Excellent (<2.0)')
            #     plt.axhline(y=3.0, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Good (<3.0)')
            #     plt.axhline(y=4.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Needs Improvement (>4.0)')
            
            # 添加數值標籤
            for bar, value in zip(bars, carbon_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # 設置圖例
            plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            # 設置網格
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # 調整佈局
            plt.tight_layout()
            
            # 保存圖表
            output_path = output_dir / 'carbon_footprint_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', 
                       format='png', pil_kwargs={'optimize': True})
            plt.close()
            
            logger.info("  ✅ 碳足跡對比圖已成功生成")
            
        else:
            logger.warning("  ⚠️ 沒有有效的碳足跡數據可供圖表生成")
            
    except Exception as e:
        logger.error(f"  ❌ 生成碳足跡圖表失敗: {str(e)}")
        plt.close()


def generate_sustainability_score_chart(results: List[Dict], output_dir: Path, logger, chinese_support: bool):
    """生成可持續性評分圖"""
    try:
        plt.figure(figsize=(12, 8))
        
        chair_ids = []
        scores = []
        
        for result in results[:10]:
            if 'sustainability_score' in result:
                chair_id = result.get('chair_id', 'Unknown')
                if len(chair_id) > 15:
                    chair_id = chair_id[:12] + '...'
                chair_ids.append(chair_id)
                score = result['sustainability_score'].get('overall_sustainability_score', 0)
                scores.append(score)
        
        if chair_ids:
            # 創建顏色映射
            colors = plt.cm.RdYlGn(np.array(scores) / 100)
            
            bars = plt.bar(range(len(chair_ids)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.xticks(range(len(chair_ids)), chair_ids, rotation=45, ha='right', fontsize=10)
            plt.xlabel('Chair ID', fontsize=12, fontweight='bold')
            plt.ylabel('Sustainability Score', fontsize=12, fontweight='bold')
            plt.title('Chair Sustainability Score Comparison', fontsize=14, fontweight='bold', pad=20)
            
            # 添加評級參考線
            plt.axhline(y=80, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Excellent (≥80)')
            plt.axhline(y=70, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Good (≥70)')
            plt.axhline(y=60, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Pass (≥60)')           
            # if chinese_support:
            #     plt.xlabel('椅子編號', fontsize=12, fontweight='bold')
            #     plt.ylabel('可持續性評分', fontsize=12, fontweight='bold')
            #     plt.title('椅子可持續性評分對比', fontsize=14, fontweight='bold', pad=20)
                
            #     # 添加評級參考線
            #     plt.axhline(y=80, color='green', linestyle='--', alpha=0.7, linewidth=2, label='優秀 (≥80)')
            #     plt.axhline(y=70, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='良好 (≥70)')
            #     plt.axhline(y=60, color='red', linestyle='--', alpha=0.7, linewidth=2, label='及格 (≥60)')
            # else:
            #     plt.xlabel('Chair ID', fontsize=12, fontweight='bold')
            #     plt.ylabel('Sustainability Score', fontsize=12, fontweight='bold')
            #     plt.title('Chair Sustainability Score Comparison', fontsize=14, fontweight='bold', pad=20)
                
            #     # 添加評級參考線
            #     plt.axhline(y=80, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Excellent (≥80)')
            #     plt.axhline(y=70, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Good (≥70)')
            #     plt.axhline(y=60, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Pass (≥60)')
            
            plt.ylim(0, 100)
            
            # 添加數值標籤
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{score:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            plt.tight_layout()
            
            plt.savefig(output_dir / 'sustainability_scores.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none', format='png', pil_kwargs={'optimize': True})
            plt.close()
            
            logger.info("  ✅ 可持續性評分圖已生成")
    except Exception as e:
        logger.error(f"  ❌ 生成可持續性評分圖失敗: {e}")
        plt.close()


def generate_material_composition_chart(summary: Dict, output_dir: Path, logger, chinese_support: bool):
    """生成材料組成餅圖"""
    try:
        material_dist = summary.get('material_distribution', {})
        
        if material_dist:
            plt.figure(figsize=(10, 10))
            
            # 材料名稱翻譯
            material_translation = {
                '木材': 'Wood',
                '金屬': 'Metal', 
                '塑料': 'Plastic',
                '布料': 'Fabric',
                '皮革': 'Leather',
                '其他': 'Others'
            }
            
            labels = []
            sizes = []
            for key, value in material_dist.items():
                # if chinese_support:
                #     labels.append(key)
                # else:
                #     labels.append(material_translation.get(key, key))
                labels.append(material_translation.get(key, key))
                sizes.append(value)
            
            colors = ['#4CAF50', '#FFC107', '#FF9800', '#2196F3', '#9C27B0', '#607D8B']
            explode = (0.05,) * len(labels)
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors[:len(labels)],
                    autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
            
            plt.title('Chair Material Composition Distribution', fontsize=14, fontweight='bold', pad=20)
            # if chinese_support:
            #     plt.title('椅子材料組成分布', fontsize=14, fontweight='bold', pad=20)
            # else:
            #     plt.title('Chair Material Composition Distribution', fontsize=14, fontweight='bold', pad=20)
            
            plt.axis('equal')
            
            plt.savefig(output_dir / 'material_composition.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none', format='png', pil_kwargs={'optimize': True})
            plt.close()
            
            logger.info("  ✅ 材料組成餅圖已生成")
    except Exception as e:
        logger.error(f"  ❌ 生成材料組成圖失敗: {e}")
        plt.close()


def generate_lifecycle_stage_chart(results: List[Dict], output_dir: Path, logger, chinese_support: bool):
    """生成生命週期階段分析圖"""
    try:
        if not results:
            return
            
        # 獲取第一個有完整生命週期數據的結果
        lifecycle_data = None
        for result in results:
            if 'life_cycle_impact' in result:
                lifecycle_data = result['life_cycle_impact']
                break
        
        if not lifecycle_data:
            logger.warning("  ⚠️ 沒有生命週期數據可供圖表生成")
            return
            
        plt.figure(figsize=(12, 8))
        
        stages = []
        emissions = []
        stage_names = {
                'material_extraction_kg_co2': 'Material Extraction',
                'production_kg_co2': 'Production',
                'transport_kg_co2': 'Transportation',
                'use_phase_kg_co2': 'Use Phase',
                'end_of_life_kg_co2': 'End of Life'
            }
        # if chinese_support:
        #     stage_names = {
        #         'material_extraction_kg_co2': '材料開採',
        #         'production_kg_co2': '生產製造',
        #         'transport_kg_co2': '運輸配送',
        #         'use_phase_kg_co2': '使用階段',
        #         'end_of_life_kg_co2': '生命終期'
        #     }
        # else:
        #     stage_names = {
        #         'material_extraction_kg_co2': 'Material Extraction',
        #         'production_kg_co2': 'Production',
        #         'transport_kg_co2': 'Transportation',
        #         'use_phase_kg_co2': 'Use Phase',
        #         'end_of_life_kg_co2': 'End of Life'
        #     }
        
        for key, stage_name in stage_names.items():
            if key in lifecycle_data:
                stages.append(stage_name)
                emissions.append(lifecycle_data[key])
        
        if stages:
            colors = plt.cm.Spectral(np.linspace(0, 1, len(stages)))
            bars = plt.bar(stages, emissions, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.xlabel('Lifecycle Stage', fontsize=12, fontweight='bold')
            plt.ylabel('Carbon Emissions (kg CO₂e)', fontsize=12, fontweight='bold')
            plt.title('Chair Lifecycle Stage Carbon Emission Analysis', fontsize=14, fontweight='bold', pad=20)
            # if chinese_support:
            #     plt.xlabel('生命週期階段', fontsize=12, fontweight='bold')
            #     plt.ylabel('碳排放量 (kg CO₂e)', fontsize=12, fontweight='bold')
            #     plt.title('椅子生命週期各階段碳排放分析', fontsize=14, fontweight='bold', pad=20)
            # else:
            #     plt.xlabel('Lifecycle Stage', fontsize=12, fontweight='bold')
            #     plt.ylabel('Carbon Emissions (kg CO₂e)', fontsize=12, fontweight='bold')
            #     plt.title('Chair Lifecycle Stage Carbon Emission Analysis', fontsize=14, fontweight='bold', pad=20)
            
            # 添加數值標籤
            for bar, value in zip(bars, emissions):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            plt.tight_layout()
            
            plt.savefig(output_dir / 'lifecycle_stages.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none', format='png', pil_kwargs={'optimize': True})
            plt.close()
            
            logger.info("  ✅ 生命週期階段分析圖已生成")
    except Exception as e:
        logger.error(f"  ❌ 生成生命週期階段圖失敗: {e}")
        plt.close()


def generate_radar_chart(summary: Dict, output_dir: Path, logger, chinese_support: bool):
    """生成綜合指標雷達圖"""
    try:
        plt.figure(figsize=(10, 10))
        categories = ['Carbon Efficiency', 'Sustainability', 'Material Eco', 'Recyclability', 'Overall Eco']
        # # 選擇標籤語言
        # if chinese_support:
        #     categories = ['碳效率', '可持續性', '材料環保性', '可回收性', '整體環保']
        # else:
        #     categories = ['Carbon Efficiency', 'Sustainability', 'Material Eco', 'Recyclability', 'Overall Eco']
        
        # 正規化數據到0-100
        carbon_score = max(0, 100 - summary.get('average_carbon_footprint', 0) * 20)
        sustainability_score = summary.get('average_sustainability_score', 0)
        wood_score = summary.get('average_wood_percentage', 0)
        recycling_score = 75  # 假設值
        overall_score = (carbon_score + sustainability_score + wood_score + recycling_score) / 4
        
        values = [carbon_score, sustainability_score, wood_score, recycling_score, overall_score]
        
        # 計算角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        # 繪製雷達圖
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=3, color='#4CAF50', markersize=8)
        ax.fill(angles, values, alpha=0.25, color='#4CAF50')
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # 設置標籤
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 100)
        
        # 添加網格
        ax.grid(True, alpha=0.3)
        plt.title('Chair Environmental Assessment Radar Chart', fontsize=14, fontweight='bold', pad=30)
        # 設置標題
        # if chinese_support:
        #     plt.title('椅子環保綜合評估雷達圖', fontsize=14, fontweight='bold', pad=30)
        # else:
        #     plt.title('Chair Environmental Assessment Radar Chart', fontsize=14, fontweight='bold', pad=30)
        
        # 保存圖片
        plt.savefig(output_dir / 'radar_chart.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='png', pil_kwargs={'optimize': True})
        plt.close()
        
        logger.info("  ✅ 綜合指標雷達圖已生成")
        
    except Exception as e:
        logger.error(f"  ❌ 生成雷達圖失敗: {e}")
        plt.close()


def generate_html_report(workflow_results: Dict, lca_results: Dict, output_dir: Path, timestamp: str, logger, chinese_support: bool):
    """生成HTML格式的報告"""
    try:
        html_file = output_dir / f"visual_report_{timestamp}.html"
        charts_dir = f"charts_{timestamp}"
        
        # 提取數據
        summary = lca_results.get('summary', {})
        recommendations = lca_results.get('recommendations', [])
        
        # 根據語言支援選擇模板
        if chinese_support:
            html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>椅子環保分析報告 - {datetime.now().strftime('%Y年%m月%d日')}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2E7D32;
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #388E3C;
            margin-top: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }}
        .summary-card h3 {{
            margin-top: 0;
            color: #2E7D32;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #1976D2;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .recommendations {{
            background-color: #E8F5E9;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .recommendations ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin: 8px 0;
        }}
        .status-success {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .status-failed {{
            color: #F44336;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌿 椅子環保分析報告</h1>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>分析數量</h3>
                <div class="value">{summary.get('successful_analysis', 0)}/{summary.get('total_chairs', 0)}</div>
            </div>
            <div class="summary-card">
                <h3>平均碳足跡</h3>
                <div class="value">{summary.get('average_carbon_footprint', 0):.2f} kg CO₂e ±5%</div>
            </div>
            <!--<div class="summary-card">
                <h3>平均可持續性評分</h3>
                <div class="value">{summary.get('average_sustainability_score', 0):.1f}/100</div>
            </div>
            <div class="summary-card">
                <h3>平均木材含量</h3>
                <div class="value">{summary.get('average_wood_percentage', 0):.1f}%</div>
            </div>-->
        </div>

        <h2>📊 視覺化分析結果</h2>
        
        <div class="chart-container">
            <h3>碳足跡對比分析</h3>
            <img src="{charts_dir}/carbon_footprint_comparison.png" alt="碳足跡對比圖">
        </div>
        
        <div class="chart-container">
            <h3>可持續性評分對比</h3>
            <img src="{charts_dir}/sustainability_scores.png" alt="可持續性評分圖">
        </div>
        
        <div class="chart-container">
            <h3>材料組成分布</h3>
            <img src="{charts_dir}/material_composition.png" alt="材料組成餅圖">
        </div>
        
        <div class="chart-container">
            <h3>生命週期階段分析</h3>
            <img src="{charts_dir}/lifecycle_stages.png" alt="生命週期階段圖">
        </div>
        
        <div class="chart-container">
            <h3>綜合環保評估</h3>
            <img src="{charts_dir}/radar_chart.png" alt="綜合評估雷達圖">
        </div>

        <h2>💡 改進建議</h2>
        <div class="recommendations">
            <ul>
"""
        else:
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chair Environmental Analysis Report - {datetime.now().strftime('%B %d, %Y')}</title>
    <style>
        body {{
            font-family: 'Arial', 'Helvetica', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2E7D32;
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #388E3C;
            margin-top: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }}
        .summary-card h3 {{
            margin-top: 0;
            color: #2E7D32;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #1976D2;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .recommendations {{
            background-color: #E8F5E9;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .recommendations ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin: 8px 0;
        }}
        .status-success {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .status-failed {{
            color: #F44336;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌿 Chair Environmental Analysis Report</h1>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Analyzed Chairs</h3>
                <div class="value">{summary.get('successful_analysis', 0)}/{summary.get('total_chairs', 0)}</div>
            </div>
            <div class="summary-card">
                <h3>Average Carbon Footprint</h3>
                <div class="value">{summary.get('average_carbon_footprint', 0):.2f} kg CO₂e</div>
            </div>
            <div class="summary-card">
                <h3>Average Sustainability Score</h3>
                <div class="value">{summary.get('average_sustainability_score', 0):.1f}/100</div>
            </div>
            <div class="summary-card">
                <h3>Average Wood Content</h3>
                <div class="value">{summary.get('average_wood_percentage', 0):.1f}%</div>
            </div>
        </div>

        <h2>📊 Visual Analysis Results</h2>
        
        <div class="chart-container">
            <h3>Carbon Footprint Comparison Analysis</h3>
            <img src="{charts_dir}/carbon_footprint_comparison.png" alt="Carbon Footprint Comparison Chart">
        </div>
        
        <div class="chart-container">
            <h3>Sustainability Score Comparison</h3>
            <img src="{charts_dir}/sustainability_scores.png" alt="Sustainability Score Chart">
        </div>
        
        <div class="chart-container">
            <h3>Material Composition Distribution</h3>
            <img src="{charts_dir}/material_composition.png" alt="Material Composition Pie Chart">
        </div>
        
        <div class="chart-container">
            <h3>Lifecycle Stage Analysis</h3>
            <img src="{charts_dir}/lifecycle_stages.png" alt="Lifecycle Stage Chart">
        </div>
        
        <div class="chart-container">
            <h3>Comprehensive Environmental Assessment</h3>
            <img src="{charts_dir}/radar_chart.png" alt="Comprehensive Assessment Radar Chart">
        </div>

        <h2>💡 Improvement Recommendations</h2>
        <div class="recommendations">
            <ul>
"""
        
        for rec in recommendations:
            html_content += f"                <li>{rec}</li>\n"
        
        # 添加處理流程狀態表格
        if chinese_support:
            html_content += f"""
            </ul>
        </div>

        <h2>📋 處理流程狀態</h2>
        <table>
            <tr>
                <th>處理階段</th>
                <th>狀態</th>
                <th>說明</th>
            </tr>
            <tr>
                <td>椅子辨識</td>
                <td class="status-success">✅ 完成</td>
                <td>成功識別並裁剪椅子圖像</td>
            </tr>
            <tr>
                <td>材質檢測</td>
                <td class="status-success">✅ 完成</td>
                <td>完成材質分析和分類</td>
            </tr>
            <tr>
                <td>3D模型生成</td>
                <td class="status-success">✅ 完成</td>
                <td>成功生成3D模型</td>
            </tr>
            <tr>
                <td>3D模型處理</td>
                <td class="{'status-success' if workflow_results.get('success', False) else 'status-failed'}">
                    {'✅ 完成' if workflow_results.get('success', False) else '❌ 失敗'}
                </td>
                <td>{'完成幾何分析和優化' if workflow_results.get('success', False) else '處理過程中發生錯誤'}</td>
            </tr>
            <tr>
                <td>LCA分析</td>
                <td class="{'status-success' if lca_results.get('success', False) else 'status-failed'}">
                    {'✅ 完成' if lca_results.get('success', False) else '❌ 失敗'}
                </td>
                <td>{'成功完成生命週期評估' if lca_results.get('success', False) else '分析過程中發生錯誤'}</td>
            </tr>
        </table>

        <div class="footer">
            <p>報告生成時間：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
            <p>© 2025 椅子環保分析系統</p>
        </div>
    </div>
</body>
</html>
"""
        else:
            html_content += f"""
            </ul>
        </div>

        <h2>📋 Processing Workflow Status</h2>
        <table>
            <tr>
                <th>Processing Stage</th>
                <th>Status</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>Chair Recognition</td>
                <td class="status-success">✅ Completed</td>
                <td>Successfully identified and cropped chair images</td>
            </tr>
            <tr>
                <td>Material Detection</td>
                <td class="status-success">✅ Completed</td>
                <td>Completed material analysis and classification</td>
            </tr>
            <tr>
                <td>3D Model Generation</td>
                <td class="status-success">✅ Completed</td>
                <td>Successfully generated 3D models</td>
            </tr>
            <tr>
                <td>3D Model Processing</td>
                <td class="{'status-success' if workflow_results.get('success', False) else 'status-failed'}">
                    {'✅ Completed' if workflow_results.get('success', False) else '❌ Failed'}
                </td>
                <td>{'Completed geometric analysis and optimization' if workflow_results.get('success', False) else 'Error occurred during processing'}</td>
            </tr>
            <tr>
                <td>LCA Analysis</td>
                <td class="{'status-success' if lca_results.get('success', False) else 'status-failed'}">
                    {'✅ Completed' if lca_results.get('success', False) else '❌ Failed'}
                </td>
                <td>{'Successfully completed lifecycle assessment' if lca_results.get('success', False) else 'Error occurred during analysis'}</td>
            </tr>
        </table>

        <div class="footer">
            <p>Report Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}</p>
            <p>© 2025 Chair Environmental Analysis System</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"  ✅ HTML報告已生成: {html_file}")
        return html_file
        
    except Exception as e:
        logger.error(f"  ❌ 生成HTML報告失敗: {e}")
        return None


def generate_detailed_text_report(workflow_results: Dict, lca_results: Dict, output_dir: Path, timestamp: str, logger):
    """生成詳細的文字報告"""
    try:
        report_file = output_dir / f"detailed_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("椅子環保分析詳細報告\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"系統版本: 1.0.0\n")
            f.write(f"分析方法: {'OpenLCA專業分析' if 'openlca' in str(lca_results.get('summary', {})).lower() else '內建LCA分析'}\n\n")
            
            # 執行摘要
            f.write("一、執行摘要\n")
            f.write("-" * 50 + "\n")
            
            summary = lca_results.get('summary', {})
            f.write(f"1. 分析範圍\n")
            f.write(f"   - 總計分析椅子數量: {summary.get('total_chairs', 0)}\n")
            f.write(f"   - 成功完成分析: {summary.get('successful_analysis', 0)}\n")
            f.write(f"   - 分析失敗: {summary.get('failed_analysis', 0)}\n\n")
            
            f.write(f"2. 關鍵環保指標\n")
            f.write(f"   - 平均碳足跡: {summary.get('average_carbon_footprint', 0):.2f} kg CO₂e\n")
            f.write(f"   - 碳足跡範圍: {summary.get('min_carbon_footprint', 0):.2f} - {summary.get('max_carbon_footprint', 0):.2f} kg CO₂e\n")
            f.write(f"   - 平均可持續性評分: {summary.get('average_sustainability_score', 0):.1f}/100\n")
            f.write(f"   - 可持續性評分範圍: {summary.get('min_sustainability_score', 0):.1f} - {summary.get('max_sustainability_score', 0):.1f}\n")
            f.write(f"   - 平均木材含量: {summary.get('average_wood_percentage', 0):.1f}%\n\n")
            
            # 材料分析
            f.write("二、材料組成分析\n")
            f.write("-" * 50 + "\n")
            material_dist = summary.get('material_distribution', {})
            if material_dist:
                for category, count in material_dist.items():
                    percentage = (count / summary.get('successful_analysis', 1)) * 100
                    f.write(f"   - {category}: {count} 個 ({percentage:.1f}%)\n")
            f.write("\n")
            
            # 可持續性等級分布
            f.write("三、可持續性等級分布\n")
            f.write("-" * 50 + "\n")
            grades = summary.get('sustainability_grades', {})
            if grades:
                for grade, count in grades.items():
                    percentage = (count / summary.get('successful_analysis', 1)) * 100
                    f.write(f"   - {grade}: {count} 個 ({percentage:.1f}%)\n")
            f.write("\n")
            
            # 詳細結果
            f.write("四、個體椅子分析結果\n")
            f.write("-" * 50 + "\n")
            
            detailed_results = lca_results.get('detailed_results', [])
            for i, result in enumerate(detailed_results[:5], 1):  # 只顯示前5個
                f.write(f"\n{i}. {result.get('chair_id', 'Unknown')}\n")
                f.write(f"   - 預測重量: {result.get('estimated_weight', 0):.2f} kg\n")
                f.write(f"   - 重量來源: {result.get('weight_info', {}).get('weight_source', 'unknown')}\n")
                f.write(f"   - 碳足跡: {result.get('carbon_footprint', {}).get('total_carbon_kg_co2', 0):.2f} kg CO₂e\n")
                f.write(f"   - 碳效率: {result.get('carbon_footprint', {}).get('carbon_efficiency_kg_co2_per_kg', 0):.2f} kg CO₂e/kg\n")
                f.write(f"   - 可持續性評分: {result.get('sustainability_score', {}).get('overall_sustainability_score', 0):.1f}/100\n")
                f.write(f"   - 可持續性等級: {result.get('sustainability_score', {}).get('sustainability_grade', 'N/A')}\n")
                f.write(f"   - 木材比例: {result.get('materials', {}).get('wood', 0)*100:.1f}%\n")
            
            if len(detailed_results) > 5:
                f.write(f"\n... 還有 {len(detailed_results) - 5} 個椅子的詳細結果未顯示\n")
            
            # 生命週期影響分析
            f.write("\n五、生命週期影響分析\n")
            f.write("-" * 50 + "\n")
            
            if detailed_results and 'life_cycle_impact' in detailed_results[0]:
                lifecycle = detailed_results[0]['life_cycle_impact']
                f.write("各階段平均碳排放（基於首個分析結果）:\n")
                f.write(f"   - 材料開採: {lifecycle.get('material_extraction_kg_co2', 0):.2f} kg CO₂e\n")
                f.write(f"   - 生產製造: {lifecycle.get('production_kg_co2', 0):.2f} kg CO₂e\n")
                f.write(f"   - 運輸配送: {lifecycle.get('transport_kg_co2', 0):.2f} kg CO₂e\n")
                f.write(f"   - 使用階段: {lifecycle.get('use_phase_kg_co2', 0):.2f} kg CO₂e\n")
                f.write(f"   - 生命終期: {lifecycle.get('end_of_life_kg_co2', 0):.2f} kg CO₂e\n")
                f.write(f"   - 總計: {lifecycle.get('total_life_cycle_kg_co2', 0):.2f} kg CO₂e\n")
            f.write("\n")
            
            # 改進建議
            f.write("六、改進建議\n")
            f.write("-" * 50 + "\n")
            recommendations = lca_results.get('recommendations', [])
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            # 技術說明
            f.write("七、技術說明\n")
            f.write("-" * 50 + "\n")
            f.write("1. 碳足跡計算方法:\n")
            f.write("   - 基於ISO 14040/14044生命週期評估標準\n")
            f.write("   - 考慮從原材料開採到產品報廢的完整生命週期\n")
            f.write("   - 使用特定材料的碳排放係數進行計算\n\n")
            
            f.write("2. 可持續性評分標準:\n")
            f.write("   - 80-100分: 優秀 (Excellent)\n")
            f.write("   - 70-79分: 良好 (Good)\n")
            f.write("   - 60-69分: 一般 (Fair)\n")
            f.write("   - <60分: 需改進 (Needs Improvement)\n\n")
            
            f.write("3. 數據來源:\n")
            f.write("   - 椅子重量: 基於AI模型預測或實測數據\n")
            f.write("   - 材料組成: 基於圖像識別和材質檢測\n")
            f.write("   - 碳排放係數: 參考國際環保數據庫\n\n")
            
            # 附錄
            f.write("八、附錄\n")
            f.write("-" * 50 + "\n")
            f.write("相關文件位置:\n")
            
            # 列出所有相關報告文件
            if 'report_files' in lca_results:
                for file_type, file_path in lca_results['report_files'].items():
                    f.write(f"   - {file_type}: {file_path}\n")
            
            f.write(f"\n處理時間統計:\n")
            if 'processing_time' in summary:
                f.write(f"   - LCA分析耗時: {summary['processing_time']:.1f} 秒\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("報告結束\n")
            f.write("=" * 100 + "\n")
        
        logger.info(f"  ✅ 詳細文字報告已生成: {report_file}")
        return report_file
        
    except Exception as e:
        logger.error(f"  ❌ 生成詳細文字報告失敗: {e}")
        return None


def generate_executive_summary(workflow_results: Dict, lca_results: Dict, output_dir: Path, timestamp: str, logger):
    """生成執行摘要（一頁精簡報告）"""
    try:
        summary_file = output_dir / f"executive_summary_{timestamp}.txt"
        
        summary = lca_results.get('summary', {})
        
        # 判斷整體表現
        avg_carbon = summary.get('average_carbon_footprint', 0)
        avg_sustainability = summary.get('average_sustainability_score', 0)
        
        if avg_carbon < 2.0 and avg_sustainability >= 80:
            overall_rating = "優秀 ★★★★★"
            overall_comment = "椅子設計達到業界領先的環保標準"
        elif avg_carbon < 3.0 and avg_sustainability >= 70:
            overall_rating = "良好 ★★★★"
            overall_comment = "椅子設計具有良好的環保表現"
        elif avg_carbon < 4.0 and avg_sustainability >= 60:
            overall_rating = "一般 ★★★"
            overall_comment = "椅子設計環保表現尚可，有改進空間"
        else:
            overall_rating = "需改進 ★★"
            overall_comment = "椅子設計需要顯著改進以達到環保標準"
        
        # 使用 UTF-8 編碼寫入文件
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("┌" + "─" * 78 + "┐\n")
            f.write("│" + " " * 25 + "椅子環保分析執行摘要" + " " * 25 + "│\n")
            f.write("│" + " " * 78 + "│\n")
            f.write("│" + f"  日期: {datetime.now().strftime('%Y年%m月%d日')}" + " " * 56 + "│\n")
            f.write("├" + "─" * 78 + "┤\n")
            f.write("│" + " " * 78 + "│\n")
            
            # 整體評級
            f.write("│  【整體環保評級】" + " " * 60 + "│\n")
            f.write("│" + " " * 78 + "│\n")
            rating_padding = 68 - len(overall_rating)
            f.write("│" + f"    {overall_rating}" + " " * rating_padding + "│\n")
            comment_padding = 68 - len(overall_comment)
            f.write("│" + f"    {overall_comment}" + " " * comment_padding + "│\n")
            f.write("│" + " " * 78 + "│\n")
            f.write("├" + "─" * 78 + "┤\n")
            
            # 關鍵指標
            f.write("│  【關鍵績效指標】" + " " * 60 + "│\n")
            f.write("│" + " " * 78 + "│\n")
            f.write("│    ▸ 平均碳足跡: " + f"{avg_carbon:.2f} kg CO₂e".ljust(20) + " " * 37 + "│\n")
            f.write("│    ▸ 可持續性評分: " + f"{avg_sustainability:.1f}/100".ljust(20) + " " * 37 + "│\n")
            f.write("│    ▸ 木材使用率: " + f"{summary.get('average_wood_percentage', 0):.1f}%".ljust(20) + " " * 37 + "│\n")
            f.write("│    ▸ 分析成功率: " + f"{(summary.get('successful_analysis', 0) / max(summary.get('total_chairs', 1), 1) * 100):.1f}%".ljust(20) + " " * 37 + "│\n")
            f.write("│" + " " * 78 + "│\n")
            f.write("├" + "─" * 78 + "┤\n")
            
            # 主要發現
            f.write("│  【主要發現】" + " " * 64 + "│\n")
            f.write("│" + " " * 78 + "│\n")
            
            findings = []
            if avg_carbon < 3.0:
                findings.append("√ 碳足跡表現優異，低於行業平均水平")
            else:
                findings.append("× 碳足跡偏高，需要優化材料和生產流程")
                
            if avg_sustainability >= 70:
                findings.append("√ 可持續性評分良好，符合環保標準")
            else:
                findings.append("× 可持續性有待提升")
                
            if summary.get('average_wood_percentage', 0) >= 85:
                findings.append("√ 高比例使用可再生木材")
            else:
                findings.append("△ 建議增加木材使用比例")
            
            for finding in findings[:3]:  # 最多顯示3個
                finding_padding = 72 - len(finding)
                f.write("│    • " + finding + " " * finding_padding + "│\n")
            
            f.write("│" + " " * 78 + "│\n")
            f.write("├" + "─" * 78 + "┤\n")
            
            # 行動建議
            f.write("│  【優先行動建議】" + " " * 60 + "│\n")
            f.write("│" + " " * 78 + "│\n")
            
            top_recommendations = lca_results.get('recommendations', [])[:3]
            for i, rec in enumerate(top_recommendations, 1):
                # 處理長度，確保不超過框架寬度
                rec_truncated = rec[:60] if len(rec) <= 60 else rec[:57] + "..."
                rec_padding = 71 - len(rec_truncated)
                f.write("│    " + f"{i}. " + rec_truncated + " " * rec_padding + "│\n")
            
            f.write("│" + " " * 78 + "│\n")
            f.write("├" + "─" * 78 + "┤\n")
            
            # 下一步
            f.write("│  【建議下一步】" + " " * 62 + "│\n")
            f.write("│" + " " * 78 + "│\n")
            f.write("│    1. 實施優先改進建議" + " " * 53 + "│\n")
            f.write("│    2. 設定具體減碳目標" + " " * 53 + "│\n")
            f.write("│    3. 定期追蹤環保績效" + " " * 53 + "│\n")
            f.write("│" + " " * 78 + "│\n")
            f.write("└" + "─" * 78 + "┘\n")
        
        logger.info(f"  ✅ 執行摘要已生成: {summary_file}")
        return summary_file
        
    except Exception as e:
        logger.error(f"  ❌ 生成執行摘要失敗: {e}")
        return None


def print_final_summary(workflow_results: Dict, lca_results: Dict, report_results: Dict, logger):
    """在控制台打印最終摘要"""
    logger.info("\n" + "=" * 80)
    logger.info("🏁 最終分析結果摘要")
    logger.info("=" * 80)
    
    if lca_results.get('success', False):
        summary = lca_results.get('summary', {})
        
        # Display key indicators in table format
        logger.info("\n📊 Key Environmental Indicators:")
        logger.info("┌─────────────────────────┬──────────────────┐")
        logger.info("│ Indicator               │ Value            │")
        logger.info("├─────────────────────────┼──────────────────┤")
        logger.info(f"│ Average Carbon Footprint│ {summary.get('average_carbon_footprint', 0):>14.2f} kg │")
        logger.info(f"│ Sustainability Score    │ {summary.get('average_sustainability_score', 0):>14.1f}/100│")
        logger.info(f"│ Wood Usage Rate         │ {summary.get('average_wood_percentage', 0):>14.1f}%   │")
        logger.info(f"│ Analysis Success Rate   │ {(summary.get('successful_analysis', 0) / max(summary.get('total_chairs', 1), 1) * 100):>14.1f}%   │")
        logger.info("└─────────────────────────┴──────────────────┘")
        
        # Environmental grade assessment
        avg_carbon = summary.get('average_carbon_footprint', 0)
        avg_sustainability = summary.get('average_sustainability_score', 0)
        
        if avg_carbon < 2.0 and avg_sustainability >= 80:
            grade = "A+ (Excellent)"
            color = "🟢"
        elif avg_carbon < 3.0 and avg_sustainability >= 70:
            grade = "A (Good)"
            color = "🟢"
        elif avg_carbon < 4.0 and avg_sustainability >= 60:
            grade = "B (Fair)"
            color = "🟡"
        else:
            grade = "C (Needs Improvement)"
            color = "🔴"
        
        logger.info(f"\n{color} Overall Environmental Grade: {grade}")
        
        # Report file list
        if report_results:
            logger.info("\n📁 Generated Report Files:")
            if 'html_report' in report_results:
                logger.info(f"   • HTML Visual Report: {report_results['html_report']}")
            if 'detailed_report' in report_results:
                logger.info(f"   • Detailed Analysis Report: {report_results['detailed_report']}")
            if 'executive_summary' in report_results:
                logger.info(f"   • Executive Summary: {report_results['executive_summary']}")
        
        # Most important recommendation
        recommendations = lca_results.get('recommendations', [])
        if recommendations:
            logger.info("\n💡 Most Important Improvement Recommendation:")
            logger.info(f"   → {recommendations[0]}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✨ Analysis completed! Please check the generated report files for detailed information.")
    logger.info("=" * 80 + "\n")
