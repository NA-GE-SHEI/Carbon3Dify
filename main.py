import subprocess, os, sys, time
import logging, json
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

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

# 設置主日誌記錄器
logger = logging.getLogger(__name__)

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
        
        # 構建命令
        cmd = [sys.executable, workflow_path]
        
        # 添加配置參數
        if workflow_config:
            # 如果有配置，先保存為臨時配置文件
            temp_config_file = "temp_workflow_config.json"
            with open(temp_config_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_config, f, indent=2, ensure_ascii=False)
            cmd.extend(["--config", temp_config_file])
        
        # 執行工作流程
        logger.info(f"執行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, text=True, capture_output=True)
        
        if result.returncode == 0:
            logger.info("✅ 工作流程執行成功")
            logger.info("工作流程輸出:")
            logger.info(result.stdout)
            
            # 清理臨時配置文件
            if workflow_config and os.path.exists("temp_workflow_config.json"):
                os.remove("temp_workflow_config.json")
            
            return True
        else:
            logger.error("❌ 工作流程執行失敗")
            logger.error(f"返回碼: {result.returncode}")
            logger.error(f"錯誤輸出: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 執行工作流程時發生異常: {e}")
        return False

def wait_and_check_trellis_completion(max_wait_time=3600, check_interval=30):
    """
    等待並檢查TRELLIS生成是否完成
    
    Args:
        max_wait_time: 最大等待時間（秒），默認1小時
        check_interval: 檢查間隔（秒），默認30秒
    """
    logger.info(f"等待TRELLIS生成完成，最大等待時間: {max_wait_time}秒")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        if check_3d_models_generated():
            logger.info("✅ 檢測到3D模型已生成完成")
            return True
        
        elapsed_time = time.time() - start_time
        remaining_time = max_wait_time - elapsed_time
        
        logger.info(f"等待中... 已等待: {elapsed_time:.0f}秒, 剩餘: {remaining_time:.0f}秒")
        time.sleep(check_interval)
    
    logger.warning(f"⚠️  等待超時 ({max_wait_time}秒)，將嘗試執行後續流程")
    return False

def create_workflow_config():
    """創建工作流程配置"""
    return {
        "input_dirs": {
            "glb_models": "./3d_models",
            "csv_data": "./models/chair_raw_data_v2_idmatched_revised_cleaned_aug_v2.csv"
        },
        "output_dirs": {
            "obj_models": "./obj_models", 
            "modified_obj": "./modified_obj",
            "geometry_analysis": "./geometry_analysis",
            "bootstrap_analysis": "./bootstrap_analysis",
            "enhanced_analysis": "./enhanced_analysis"
        },
        "processing": {
            "skip_existing": True,
            "generation_filter": None  # 處理所有生成變體
        },
        "bootstrap": {
            "n_iterations": 30,  # 適中的迭代次數
            "ci_level": 0.95,
            "skip_execution": False
        },
        "analysis": {
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 100,
            "max_depth": 10,
            "model_type": "all"
        },
        "chair_params": {
            "is_square": 0,
            "is_round": 1,
            "seat_area": 707,
            "seat_thickness": 3,
            "true_weight": 7.2
        },
        "enabled_steps": [1, 2, 3, 4, 5]  # 執行所有步驟
    }

def main():
    start = time.time()
    overall_success = True
    
    try:
        logger.info("🔍 開始執行椅子辨識...")
        print("Exec identifyChair...")
        identifyChair_result = subprocess.run([yolo, identifyChair_path], 
                            text=True)
        if identifyChair_result.returncode == 0:
            print("Successfully!")
            logger.info("✅ 椅子辨識完成")
        else:
            logger.error("❌ 椅子辨識失敗")
            overall_success = False
    except Exception as e:
        print(e)
        logging.error(f"椅子辨識異常: {e}")
        overall_success = False
        # 可以選擇是否繼續執行後續步驟
        # sys.exit()
    
    try:
        logger.info("🎨 開始執行材質檢測...")
        print("Exec materialDetection...")
        materialDetection_result = subprocess.run([mmsegmentation, materialDetection_path], 
                            text=True)
        if materialDetection_result.returncode == 0:
            print("Successfully!")
            logger.info("✅ 材質檢測完成")
        else:
            logger.error("❌ 材質檢測失敗")
            overall_success = False
    except Exception as e:
        print(e)
        logging.error(f"材質檢測異常: {e}")
        overall_success = False
        # 材質檢測失敗不影響後續3D生成
    
    try:
        logger.info("🏗️ 開始執行TRELLIS 3D模型生成...")
        print("Exec trellisAutoGeneration...")
        trellis_result = subprocess.run([trellis, trellis_path], 
                            text=True)
        if trellis_result.returncode == 0:
            print("Successfully!")
            logger.info("✅ TRELLIS 3D模型生成完成")
        else:
            logger.error("❌ TRELLIS 3D模型生成失敗")
            overall_success = False
            # TRELLIS失敗則無法進行後續3D處理
            logger.error("由於TRELLIS生成失敗，跳過後續3D處理流程")
            et = time.time() - start
            print(f"總耗時: {et:.2f}秒")
            return
    except Exception as e:
        print(e)
        logging.error(f"TRELLIS生成異常: {e}")
        overall_success = False
        logger.error("由於TRELLIS生成異常，跳過後續3D處理流程")
        et = time.time() - start
        print(f"總耗時: {et:.2f}秒")
        return
    
    # ======================== 新增：3D模型處理工作流程 ========================
    try:
        logger.info("⏳ 等待TRELLIS生成完成並檢查輸出...")
        
        # 等待一小段時間讓TRELLIS完全結束
        time.sleep(10)
        
        # 檢查3D模型是否生成完成
        models_ready = wait_and_check_trellis_completion(max_wait_time=600, check_interval=5)  # 等待最多10分鐘
        
        if models_ready:
            logger.info("🔄 開始執行3D模型處理與分析工作流程...")
            
            # 創建工作流程配置
            workflow_config = create_workflow_config()
            
            # 檢查CSV數據文件是否存在
            csv_file = Path(workflow_config["input_dirs"]["csv_data"])
            if not csv_file.exists():
                logger.warning(f"⚠️  CSV數據文件不存在: {csv_file}")
                logger.info("將只執行3D處理步驟（步驟1-3），跳過機器學習分析")
                workflow_config["enabled_steps"] = [1, 2, 3]  # 只執行3D處理
            
            # 執行工作流程
            workflow_success = run_workflow_integration(workflow_config)
            
            if workflow_success:
                logger.info("🎉 3D模型處理工作流程執行成功！")
                print("🎉 Complete! 3D model processing workflow finished successfully!")
            else:
                logger.error("❌ 3D模型處理工作流程執行失敗")
                overall_success = False
        else:
            logger.warning("⚠️  未檢測到3D模型生成或生成不完整，跳過後續處理")
            logger.info("您可以稍後手動執行: python workflowIntegration.py")
            
    except Exception as e:
        logger.error(f"❌ 3D模型處理工作流程異常: {e}")
        overall_success = False
    
    # ======================== 總結 ========================
    et = time.time() - start
    
    logger.info("=" * 60)
    if overall_success:
        logger.info("🎉 整個處理流程執行完成！")
        print(f"🎉 All processes completed successfully! 總耗時: {et:.2f}秒")
    else:
        logger.warning("⚠️  部分流程執行失敗，請檢查日誌")
        print(f"⚠️  Some processes failed. 總耗時: {et:.2f}秒")
    
    logger.info("=" * 60)
    logger.info("📊 處理結果摘要:")
    logger.info(f"   - 椅子辨識: {'✅' if identifyChair_result.returncode == 0 else '❌'}")
    logger.info(f"   - 材質檢測: {'✅' if 'materialDetection_result' in locals() and materialDetection_result.returncode == 0 else '❌'}")
    logger.info(f"   - 3D模型生成: {'✅' if 'trellis_result' in locals() and trellis_result.returncode == 0 else '❌'}")
    logger.info(f"   - 3D模型處理: {'✅' if 'workflow_success' in locals() and workflow_success else '❌'}")
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
        "./enhanced_analysis"
    ]
    
    for output_dir in output_dirs:
        if Path(output_dir).exists():
            file_count = len(list(Path(output_dir).rglob('*.*')))
            logger.info(f"   - {output_dir}: {file_count} 個文件")
    
    logger.info("=" * 60)

if __name__ == '__main__':
    identifyChair_path = "identifyChair.py"
    materialDetection_path = "materialDetection.py"
    trellis_path = "trellisAutoGeneration.py"
    workflow_path = "workflowIntegration.py"
    
    # 檢查所需腳本是否存在
    required_scripts = [
        identifyChair_path,
        materialDetection_path, 
        trellis_path,
        workflow_path
    ]
    
    missing_scripts = [script for script in required_scripts if not os.path.exists(script)]
    
    if missing_scripts:
        print(f"❌ 缺少必要的腳本文件: {missing_scripts}")
        logging.error(f"缺少必要的腳本文件: {missing_scripts}")
        sys.exit(1)
    
    print("🚀 開始執行完整的椅子處理流程...")
    print("流程包括: 椅子辨識 → 材質檢測 → 3D模型生成 → 3D模型處理分析")
    print("=" * 60)
    
    main()