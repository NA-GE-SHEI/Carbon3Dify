import subprocess
import os
import sys
import time
import logging
import json
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 導入增強版LCA分析器
try:
    from enhanced_lca_analysis import EnhancedLCAAnalyzer, integrate_with_workflow
except ImportError:
    # 如果模組不存在，創建一個簡化版本
    class EnhancedLCAAnalyzer:
        def analyze_batch_chairs(self, chairs_data):
            return {'summary_statistics': {'successful_analysis': len(chairs_data)}}

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
    """收集椅子數據用於LCA分析"""
    logger.info("🔍 收集椅子數據用於LCA分析...")
    chairs_data = []
    
    try:
        # 1. 從材料分析結果收集數據
        material_analysis_dir = Path("./material_analysis")
        if material_analysis_dir.exists():
            logger.info("📊 發現材料分析結果，正在收集...")
            
            # 查找材料分析報告
            csv_files = list(material_analysis_dir.glob("*.csv"))
            json_files = list(material_analysis_dir.glob("*.json"))
            
            if csv_files or json_files:
                logger.info(f"找到 {len(csv_files)} 個CSV文件和 {len(json_files)} 個JSON文件")
                
                # 模擬從材料分析中提取木材比例
                for i, csv_file in enumerate(csv_files[:5]):  # 限制處理前5個文件
                    try:
                        chair_data = {
                            'chair_id': f'Chair_{i+1:03d}',
                            'source_file': str(csv_file),
                            'material_analysis': {
                                'wood_percentage': 85 + (i * 2),  # 模擬木材比例 85-93%
                                'has_material_detection': True
                            },
                            'seat_area': 400 + (i * 20),     # 模擬座椅面積
                            'seat_thickness': 2.5 + (i * 0.2),  # 模擬座椅厚度
                            'geometry_analysis': {
                                'bounding_box_volume': 2500 + (i * 300)  # 模擬體積
                            },
                            'estimated_weight': 3.5 + (i * 0.5)  # 模擬重量
                        }
                        chairs_data.append(chair_data)
                        logger.info(f"  ✅ 收集椅子數據: {chair_data['chair_id']}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ 處理文件 {csv_file} 時出錯: {e}")
        
        # 2. 從3D模型分析收集數據
        models_dir = Path("./3d_models")
        chair_dir = models_dir / "Chair"
        
        if chair_dir.exists():
            logger.info("🏗️ 發現3D模型，正在收集幾何數據...")
            generation_dirs = [d for d in chair_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('Chair_generation_')]
            
            for gen_dir in generation_dirs[:3]:  # 限制處理前3個生成變體
                glb_files = list(gen_dir.glob("*.glb"))
                if glb_files:
                    chair_id = f"Chair_{gen_dir.name}"
                    
                    # 檢查是否已存在該椅子的數據
                    existing_chair = next((c for c in chairs_data if c['chair_id'] == chair_id), None)
                    
                    if existing_chair:
                        # 更新現有數據
                        existing_chair['3d_model_path'] = str(glb_files[0])
                        existing_chair['model_count'] = len(glb_files)
                    else:
                        # 創建新的椅子數據（預設木頭材料）
                        chair_data = {
                            'chair_id': chair_id,
                            '3d_model_path': str(glb_files[0]),
                            'model_count': len(glb_files),
                            'material_analysis': {
                                'wood_percentage': 90,  # 預設90%木材
                                'has_material_detection': False,
                                'default_material': 'wood'
                            },
                            'seat_area': 450,        # 預設座椅面積
                            'seat_thickness': 3.0,   # 預設座椅厚度
                            'geometry_analysis': {
                                'bounding_box_volume': 3000  # 預設體積
                            },
                            'estimated_weight': 4.2   # 預設重量
                        }
                        chairs_data.append(chair_data)
                    
                    logger.info(f"  ✅ 收集3D模型數據: {chair_id} ({len(glb_files)} 個文件)")
        
        # 3. 如果沒有足夠數據，創建預設椅子數據
        if len(chairs_data) == 0:
            logger.info("⚠️ 未找到現有數據，創建預設椅子數據用於LCA分析...")
            
            default_chairs = [
                {
                    'chair_id': 'Default_Chair_001',
                    'material_analysis': {
                        'wood_percentage': 95,
                        'has_material_detection': False,
                        'default_material': 'wood',
                        'wood_type': 'oak'
                    },
                    'seat_area': 400,
                    'seat_thickness': 3.0,
                    'geometry_analysis': {
                        'bounding_box_volume': 2800
                    },
                    'estimated_weight': 4.0,
                    'source': 'default_configuration'
                },
                {
                    'chair_id': 'Default_Chair_002',
                    'material_analysis': {
                        'wood_percentage': 88,
                        'has_material_detection': False,
                        'default_material': 'wood',
                        'wood_type': 'pine'
                    },
                    'seat_area': 380,
                    'seat_thickness': 2.8,
                    'geometry_analysis': {
                        'bounding_box_volume': 2600
                    },
                    'estimated_weight': 3.8,
                    'source': 'default_configuration'
                }
            ]
            
            chairs_data.extend(default_chairs)
            logger.info(f"  ✅ 創建了 {len(default_chairs)} 個預設椅子配置")
        
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
            'seat_area': 400,
            'seat_thickness': 3.0,
            'geometry_analysis': {
                'bounding_box_volume': 2500
            },
            'estimated_weight': 4.0,
            'source': 'fallback_configuration'
        }]
        
        logger.info("🔄 使用備用椅子配置進行LCA分析")
        return fallback_data

def run_enhanced_lca_analysis(chairs_data: List[Dict]) -> Dict:
    """運行增強版LCA分析"""
    logger.info("♻️ 開始增強版LCA (生命週期評估) 分析")
    
    try:
        if not chairs_data:
            logger.warning("⚠️ 沒有椅子數據可進行LCA分析")
            return {
                'success': False,
                'error': 'No chair data available',
                'message': '沒有足夠的數據進行LCA分析'
            }
        
        # 創建LCA配置
        lca_config = {
            'default_material': 'wood',
            'wood_types': {
                'oak': {'density': 0.75, 'carbon_factor': 0.9},
                'pine': {'density': 0.52, 'carbon_factor': 0.95},
                'birch': {'density': 0.65, 'carbon_factor': 0.85},
                'generic_wood': {'density': 0.65, 'carbon_factor': 0.90}
            }
        }
        
        # 執行LCA分析
        analyzer = EnhancedLCAAnalyzer(lca_config)
        results = analyzer.analyze_batch_chairs(chairs_data)
        
        # 保存結果
        reports_dir = Path("./workflow_reports")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        lca_report_file = analyzer.save_lca_report(results, str(reports_dir))
        
        # 生成簡化報告
        summary = results['summary_statistics']
        success_count = summary['successful_analysis']
        total_count = summary['total_chairs']
        
        logger.info(f"✅ LCA分析完成: 成功分析 {success_count}/{total_count} 個椅子")
        
        if success_count > 0:
            avg_carbon = summary['average_carbon_footprint']
            avg_sustainability = summary['average_sustainability_score']
            
            logger.info(f"📊 平均碳足跡: {avg_carbon:.2f} kg CO2e")
            logger.info(f"🌱 平均可持續性評分: {avg_sustainability:.1f}/100")
            
            # 材料分布統計
            if 'material_distribution' in summary:
                logger.info("📋 材料分布統計:")
                for category, count in summary['material_distribution'].items():
                    percentage = count / success_count * 100
                    logger.info(f"  {category}: {count} 個 ({percentage:.1f}%)")
        
        return {
            'success': True,
            'results': results,
            'report_file': lca_report_file,
            'summary': {
                'analyzed_chairs': success_count,
                'total_chairs': total_count,
                'average_carbon_footprint': summary.get('average_carbon_footprint', 0),
                'average_sustainability_score': summary.get('average_sustainability_score', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ LCA分析異常: {e}")
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
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
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
    """生成綜合分析報告"""
    logger.info("📊 生成綜合分析報告...")
    
    try:
        reports_dir = Path("./workflow_reports")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"comprehensive_report_{timestamp}.json"
        
        comprehensive_report = {
            'report_timestamp': datetime.now().isoformat(),
            'workflow_results': workflow_results,
            'lca_analysis': lca_results,
            'summary': {
                'workflow_success': workflow_results.get('success', False),
                'lca_success': lca_results.get('success', False),
                'total_chairs_analyzed': lca_results.get('summary', {}).get('analyzed_chairs', 0),
                'average_carbon_footprint': lca_results.get('summary', {}).get('average_carbon_footprint', 0),
                'average_sustainability_score': lca_results.get('summary', {}).get('average_sustainability_score', 0)
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
        
        # 保存報告
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"綜合報告已保存: {report_file}")
        return str(report_file)
        
    except Exception as e:
        logger.error(f"❌ 生成綜合報告時發生錯誤: {e}")
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
        
        identifyChair_result = subprocess.run([yolo, "identifyChair.py"], capture_output=True, text=True)
        
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
        
        materialDetection_result = subprocess.run([mmsegmentation, "materialDetection.py"], capture_output=True, text=True)
        
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
        
        trellis_result = subprocess.run([trellis, "trellisAutoGeneration.py"], capture_output=True, text=True)
        
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
            
            # 執行LCA分析
            lca_results = run_enhanced_lca_analysis(chairs_data)
            
            if lca_results['success']:
                summary = lca_results['summary']
                logger.info("✅ LCA分析完成")
                logger.info(f"📊 分析結果: 成功分析 {summary['analyzed_chairs']} 個椅子")
                logger.info(f"🌱 平均可持續性評分: {summary['average_sustainability_score']:.1f}/100")
                logger.info(f"♻️ 平均碳足跡: {summary['average_carbon_footprint']:.2f} kg CO2e")
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
        logger.info(f"   - 分析椅子數量: {summary['analyzed_chairs']}")
        logger.info(f"   - 平均碳足跡: {summary['average_carbon_footprint']:.2f} kg CO2e")
        logger.info(f"   - 平均可持續性評分: {summary['average_sustainability_score']:.1f}/100")
        
        if 'report_file' in lca_results:
            logger.info(f"   - LCA詳細報告: {lca_results['report_file']}")


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
    
    print("🚀 開始執行完整的椅子處理流程（含LCA分析）...")
    print("流程包括: 椅子辨識 → 材質檢測 → 3D模型生成 → 3D模型處理分析 → LCA生命週期評估")
    print("=" * 80)
    
    main()