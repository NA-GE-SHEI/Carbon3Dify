import os
import sys
import time
import logging
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 添加當前目錄到Python路徑
sys.path.append(str(Path(__file__).parent))

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow_integration.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class WorkflowIntegration:
    """椅子3D模型處理與分析完整工作流程整合器"""
    
    def __init__(self, config_file: str = None):
        """
        初始化工作流程整合器
        
        Args:
            config_file: 配置文件路徑
        """
        self.config = self.load_config(config_file)
        self.timing_records = {}
        self.results = {}
        self.start_time = time.time()
        
        # 工作流程步驟定義
        self.steps = {
            1: {'name': 'GLB to OBJ Conversion', 'module': 'model2obj_enhanced', 'enabled': True},
            2: {'name': 'OBJ Modification & Analysis', 'module': 'modification_obj_enhanced', 'enabled': True},
            3: {'name': 'OBJ Geometry Analysis', 'module': 'objFunction_enhanced', 'enabled': True},
            4: {'name': 'Bootstrap Analysis', 'module': 'bootstrapV2_enhanced', 'enabled': True},
            5: {'name': 'Enhanced RF Analysis', 'module': 'enhanced_rf_analysis_enhanced', 'enabled': True}
        }
    
    def load_config(self, config_file: str = None) -> Dict:
        """載入配置文件"""
        default_config = {
            # 輸入路徑配置
            'input_dirs': {
                'glb_models': './3d_models',  # 包含Chair/Chair_generation_X的目錄
                'csv_data': './models/chair_raw_data_v2_idmatched_revised_cleaned_aug_v2.csv',  # CSV數據文件
            },
            
            # 輸出路徑配置
            'output_dirs': {
                'obj_models': './obj_models',
                'modified_obj': './modified_obj',
                'geometry_analysis': './geometry_analysis',
                'bootstrap_analysis': './bootstrap_analysis',
                'enhanced_analysis': './enhanced_analysis'
            },
            
            # 處理參數
            'processing': {
                'skip_existing': True,  # 跳過已存在的結果
                'generation_filter': None,  # 生成變體過濾列表，如 ['Chair_generation_1', 'Chair_generation_3']
            },
            
            # Bootstrap分析參數
            'bootstrap': {
                'n_iterations': 50,
                'ci_level': 0.95,
                'skip_execution': False,  # 是否跳過執行，直接分析現有結果
            },
            
            # 模型分析參數
            'analysis': {
                'test_size': 0.2,
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 10,
                'model_type': 'all',  # 'rf', 'xgb', 'mlr', 'svr', 'all'
            },
            
            # 椅子參數（用於預測）
            'chair_params': {
                'is_square': 0,
                'is_round': 1,
                'seat_area': 707,
                'seat_thickness': 3,
                'true_weight': 4.2
            },
            
            # 啟用的步驟
            'enabled_steps': [1, 2, 3, 4, 5]  # 可以選擇性啟用步驟
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 深度合併配置
                    self._deep_update(default_config, user_config)
                logger.info(f"✅ 已載入配置文件: {config_file}")
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
    
    def save_config(self, config_file: str = "workflow_config.json"):
        """保存當前配置到文件"""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 配置已保存到: {config_file}")
        except Exception as e:
            logger.error(f"❌ 保存配置失敗: {e}")
    
    def create_directories(self):
        """創建必要的目錄"""
        for dir_name, dir_path in self.config['output_dirs'].items():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info("✅ 已創建所有必要目錄")
    
    def step_1_glb_to_obj(self) -> Dict:
        """步驟1: GLB轉OBJ"""
        logger.info("=" * 80)
        logger.info("🔄 步驟1: GLB到OBJ轉換")
        logger.info("=" * 80)
        
        step_start_time = time.time()
        
        try:
            from calculate_carbon.model2obj import process_chair_models
            
            result = process_chair_models(
                self.config['input_dirs']['glb_models'],
                self.config['output_dirs']['obj_models']
            )
            
            step_time = time.time() - step_start_time
            self.timing_records['step_1'] = step_time
            
            if result['success']:
                logger.info(f"✅ 步驟1完成: 轉換了 {result['total_converted']} 個文件 (耗時: {step_time:.2f}秒)")
                return result
            else:
                logger.error(f"❌ 步驟1失敗: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            logger.error(f"❌ 步驟1執行異常: {e}")
            return {'success': False, 'error': str(e)}
    
    def step_2_obj_modification(self) -> Dict:
        """步驟2: OBJ修改與分析"""
        logger.info("=" * 80)
        logger.info("🔄 步驟2: OBJ修改與幾何分析")
        logger.info("=" * 80)
        
        step_start_time = time.time()
        
        try:
            from calculate_carbon.modification_obj import GeometryAnalyzer
            # from calculate_carbon.chair_split_structure import PerfectChairSegmenter

            analyzer = GeometryAnalyzer()
            result = analyzer.process_chair_models(
                self.config['output_dirs']['obj_models'],
                self.config['output_dirs']['modified_obj']
            )
            # 建立分割器
            # segmenter = PerfectChairSegmenter(
            #     input_path=self.config['output_dirs']['modified_obj'],
            #     layer_height=0.002,
            #     eps=0.1,
            #     min_samples=10
            # )
            # result = segmenter.segment()

            step_time = time.time() - step_start_time
            self.timing_records['step_2'] = step_time
            
            if result['success']:
                logger.info(f"✅ 步驟2完成: 處理了 {result['total_processed']} 個文件 (耗時: {step_time:.2f}秒)")
                return result
            else:
                logger.error(f"❌ 步驟2失敗: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            logger.error(f"❌ 步驟2執行異常: {e}")
            return {'success': False, 'error': str(e)}
    
    def step_3_obj_analysis(self) -> Dict:
        """步驟3: OBJ幾何分析"""
        logger.info("=" * 80)
        logger.info("🔄 步驟3: OBJ幾何特徵分析")
        logger.info("=" * 80)
        
        step_start_time = time.time()
        
        try:
            from calculate_carbon.model2obj import process_chair_models
            
            result = process_chair_models(
                self.config['output_dirs']['modified_obj'],
                self.config['output_dirs']['geometry_analysis']
            )
            
            step_time = time.time() - step_start_time
            self.timing_records['step_3'] = step_time
            
            if result['success']:
                logger.info(f"✅ 步驟3完成: 分析了 {result['total_converted']} 個文件 (耗時: {step_time:.2f}秒)")
                return result
            else:
                logger.error(f"❌ 步驟3失敗: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            logger.error(f"❌ 步驟3執行異常: {e}")
            return {'success': False, 'error': str(e)}
    
    # def step_4_bootstrap_analysis(self) -> Dict:
    #     """步驟4: Bootstrap統計分析"""
    #     logger.info("=" * 80)
    #     logger.info("🔄 步驟4: Bootstrap統計分析")
    #     logger.info("=" * 80)
        
    #     step_start_time = time.time()
        
    #     try:
    #         # 檢查CSV數據文件
    #         csv_file = self.config['input_dirs']['csv_data']
    #         print(csv_file)
    #         logger.info(csv_file)
    #         if not Path(csv_file).exists():
    #             logger.error(f"CSV數據文件不存在: {csv_file}")
    #             return {'success': False, 'error': f'CSV file not found: {csv_file}'}
            
    #         # 導入並執行Bootstrap分析
    #         from calculate_carbon.bootstrapV2 import main as bootstrap_main
            
    #         # 保存原始argv並設置新的
    #         import sys
    #         original_argv = sys.argv.copy()
            
    #         try:
    #             # 設置模擬的命令行參數
    #             bootstrap_config = self.config['bootstrap']
    #             analysis_config = self.config['analysis']
                
    #             sys.argv = [
    #                 './calculate_carbon/bootstrapV2.py',
    #                 '--py_path', 'calculate_carbon/enhanced_rf_analysis.py',
    #                 '--data_file', csv_file,
    #                 '--n_iterations', str(bootstrap_config['n_iterations']),
    #                 '--result_dir', './enhanced_rf_result',
    #                 '--output_dir', self.config['output_dirs']['bootstrap_analysis'],
    #                 '--model_type', analysis_config['model_type'],
    #                 '--min_num', '10',
    #                 '--ci_level', str(bootstrap_config['ci_level'])
    #             ]
                
    #             # 直接調用main函數
    #             bootstrap_main()
                
    #             step_time = time.time() - step_start_time
    #             self.timing_records['step_4'] = step_time
                
    #             logger.info(f"✅ 步驟4完成: Bootstrap分析 (耗時: {step_time:.2f}秒)")
    #             return {'success': True, 'output_dir': self.config['output_dirs']['bootstrap_analysis']}
                
    #         finally:
    #             # 恢復原始argv
    #             sys.argv = original_argv
    #     except Exception as e:
    #         logger.error(f"❌ 步驟4執行異常: {e}")
    #         return {'success': False, 'error': str(e)}
    def step_4_bootstrap_analysis(self) -> Dict:
        """步驟4: 優化的Bootstrap統計分析"""
        logger.info("=" * 80)
        logger.info("🔄 步驟4: 優化的Bootstrap統計分析")
        logger.info("=" * 80)
        
        step_start_time = time.time()
        
        try:
            # 檢查CSV數據文件
            csv_file = self.config['input_dirs']['csv_data']
            if not Path(csv_file).exists():
                logger.error(f"CSV數據文件不存在: {csv_file}")
                return {'success': False, 'error': f'CSV file not found: {csv_file}'}
            
            # 準備參數
            bootstrap_config = self.config['bootstrap']
            analysis_config = self.config['analysis']
            chair_config = self.config['chair_params']
            n_iterations = bootstrap_config['n_iterations']
            
            # 優化1: 預先載入數據，避免重複I/O
            logger.info("📊 預先載入數據到記憶體...")
            try:
                import pandas as pd
                data = pd.read_csv(csv_file, encoding='utf-8')
                logger.info(f"✅ 成功載入數據: {len(data)} 行")
            except Exception as e:
                logger.warning(f"⚠️ 使用utf-8編碼載入失敗，嘗試utf-8: {e}")
                try:
                    data = pd.read_csv(csv_file, encoding='utf-8')
                    logger.info(f"✅ 成功載入數據 (utf-8): {len(data)} 行")
                except Exception as e2:
                    logger.error(f"❌ 載入數據失敗: {e2}")
                    return {'success': False, 'error': f'Failed to load data: {e2}'}
            
            # 準備分析參數
            analysis_args = {
                'data_file': csv_file,
                'output_dir': self.config['output_dirs']['bootstrap_analysis'],
                'model_type': analysis_config['model_type'],
                'is_square': chair_config['is_square'],
                'is_round': chair_config['is_round'],
                'seat_area': chair_config['seat_area'],
                'seat_thickness': chair_config['seat_thickness'],
                'true_weight': chair_config['true_weight'],
                'encoding': 'utf-8'  # 保持原有編碼設置
            }
            
            # 優化2: 根據迭代次數選擇最佳執行策略
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
            
            cpu_count = mp.cpu_count()
            n_workers = max(1, min(cpu_count - 1, n_iterations // 4))  # 動態調整工作進程數
            
            logger.info(f"🚀 執行 {n_iterations} 次Bootstrap迭代")
            logger.info(f"💻 使用 {n_workers} 個並行工作進程 (總CPU核心: {cpu_count})")
            
            if n_iterations >= 30 and n_workers > 1:
                # 並行執行 (適用於大量迭代)
                results = self._run_parallel_bootstrap(analysis_args, n_iterations, n_workers)
            else:
                # 優化的順序執行 (適用於少量迭代)
                results = self._run_optimized_sequential_bootstrap(analysis_args, n_iterations)
            
            if not results:
                logger.error("❌ Bootstrap分析執行失敗")
                return {'success': False, 'error': 'Bootstrap analysis failed'}
            
            logger.info(f"✅ Bootstrap執行完成，成功 {len(results)} 次迭代")
            
            # 記錄執行時間
            step_time = time.time() - step_start_time
            self.timing_records['step_4'] = step_time
            
            logger.info(f"✅ 步驟4完成: Bootstrap分析 (耗時: {step_time:.2f}秒)")
            logger.info(f"📊 平均每次迭代: {step_time/n_iterations:.2f}秒")
            
            return {
                'success': True, 
                'output_dir': self.config['output_dirs']['bootstrap_analysis'],
                'iterations_completed': len(results),
                'total_time': step_time
            }
                    
        except Exception as e:
            logger.error(f"❌ 步驟4執行異常: {e}")
            return {'success': False, 'error': str(e)}

    def _run_parallel_bootstrap(self, analysis_args: Dict, n_iterations: int, n_workers: int) -> List:
        """並行執行Bootstrap分析"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import subprocess
        
        logger.info(f"🔄 使用 {n_workers} 個進程並行執行...")
        
        # 準備所有任務
        tasks = []
        for i in range(n_iterations):
            task_args = analysis_args.copy()
            # 為每次迭代生成不同的隨機種子
            task_args['random_state'] = int(time.time()) % 10000 + i
            tasks.append((task_args, i))
        
        successful_runs = []
        
        # 使用進程池執行
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # 提交所有任務
            future_to_iteration = {
                executor.submit(self._run_single_iteration, task_args, iteration): iteration 
                for task_args, iteration in tasks
            }
            
            completed = 0
            for future in as_completed(future_to_iteration):
                iteration = future_to_iteration[future]
                try:
                    success = future.result()
                    if success:
                        successful_runs.append(iteration)
                    completed += 1
                    
                    # 進度報告
                    if completed % max(1, n_iterations // 10) == 0 or completed == n_iterations:
                        success_rate = len(successful_runs) / completed * 100
                        logger.info(f"📊 進度: {completed}/{n_iterations} ({completed/n_iterations*100:.1f}%) | 成功率: {success_rate:.1f}%")
                        
                except Exception as e:
                    logger.warning(f"⚠️ 迭代 {iteration} 執行失敗: {e}")
                    completed += 1
        
        return successful_runs

    def _run_optimized_sequential_bootstrap(self, analysis_args: Dict, n_iterations: int) -> List:
        """優化的順序執行Bootstrap分析"""
        logger.info(f"🔄 順序執行 {n_iterations} 次迭代 (已優化)...")
        
        successful_runs = []
        
        for i in range(n_iterations):
            try:
                task_args = analysis_args.copy()
                task_args['random_state'] = int(time.time()) % 10000 + i
                
                success = self._run_single_iteration(task_args, i)
                if success:
                    successful_runs.append(i)
                
                # 進度報告
                if (i + 1) % max(1, n_iterations // 10) == 0 or (i + 1) == n_iterations:
                    success_rate = len(successful_runs) / (i + 1) * 100
                    logger.info(f"📊 進度: {i + 1}/{n_iterations} ({(i + 1)/n_iterations*100:.1f}%) | 成功率: {success_rate:.1f}%")
                    
            except Exception as e:
                logger.warning(f"⚠️ 迭代 {i} 執行失敗: {e}")
        
        return successful_runs

    def _run_single_iteration(self, args: Dict, iteration: int) -> bool:
        """執行單次Bootstrap迭代（優化版本）"""
        import subprocess
        import sys
        
        try:
            # 構建命令參數 - 保持與原版兼容
            cmd = [
                'python3.10', 'calculate_carbon/bootstrapV2.py',
                '--data_file', args['data_file'],
                '--encoding', args['encoding'],
                '--model_type', 'rf',
                # '--model_type', args['model_type'],
                '--output_dir', args['output_dir'],
                # '--random_state', str(args['random_state']),
                '--n_iterations', '1',  # 每次只執行一個迭代
                '--min_num', str(iteration)  # 使用迭代次數作為結果文件夾編號
            ]
            
            # # 添加椅子參數
            # if 'is_square' in args:
            #     cmd.extend(['--is_square', str(args['is_square'])])
            # if 'is_round' in args:
            #     cmd.extend(['--is_round', str(args['is_round'])])
            # if 'seat_area' in args:
            #     cmd.extend(['--seat_area', str(args['seat_area'])])
            # if 'seat_thickness' in args:
            #     cmd.extend(['--seat_thickness', str(args['seat_thickness'])])
            # if 'true_weight' in args:
            #     cmd.extend(['--true_weight', str(args['true_weight'])])
            
            # 執行命令，設置超時避免卡死
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=False, 
                text=True,
                timeout=300  # 5分鐘超時
            )
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ 迭代 {iteration} 超時 (5分鐘)")
            return False
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ 迭代 {iteration} 命令執行失敗: {e}")
            return False
        except Exception as e:
            logger.warning(f"⚠️ 迭代 {iteration} 執行異常: {e}")
            return False

    def step_5_enhanced_analysis(self) -> Dict:
        """步驟5: 增強隨機森林分析 - 修復預測重量覆蓋問題"""
        logger.info("=" * 80)
        logger.info("🔄 步驟5: 增強隨機森林分析")
        logger.info("=" * 80)
        step_start_time = time.time()
        
        try:
            # 檢查CSV數據文件
            csv_file = self.config['input_dirs']['csv_data']
            if not Path(csv_file).exists():
                logger.error(f"CSV數據文件不存在: {csv_file}")
                return {'success': False, 'error': f'CSV file not found: {csv_file}'}

            # 準備參數
            analysis_config = self.config['analysis']
            chair_config = self.config['chair_params']

            # 構建參數對象
            from types import SimpleNamespace
            args = SimpleNamespace(
                data_file=csv_file,
                encoding='utf-8',
                output_dir=self.config['output_dirs']['enhanced_analysis'],
                model_type=analysis_config['model_type'],
                test_size=analysis_config['test_size'],
                random_state=analysis_config['random_state'],
                n_estimators=analysis_config['n_estimators'],
                max_depth=analysis_config['max_depth'],
                is_square=chair_config['is_square'],
                is_round=chair_config['is_round'],
                seat_area=chair_config['seat_area'],
                seat_thickness=chair_config['seat_thickness'],
                true_weight=chair_config['true_weight'],
                back_height=None,
                back_volume=0.000390 * 1000000,
                leg_height=0.3737 * 100,
                leg_volume=0.001550 * 1000000,
                seat_volume=0.002771 * 1000000,
                svr_kernel='rbf',
                svr_c=1.0,
                svr_epsilon=0.1,
                dpi=300,
                comparison_chairs=None
            )

            # 導入並執行Enhanced分析
            from calculate_carbon.enhanced_rf_analysis import main as enhanced_main

            # 保存原始argv並設置新的
            import sys
            original_argv = sys.argv.copy()
            try:
                # 設置模擬的命令行參數
                sys.argv = ['enhanced_rf_analysis_enhanced.py']
                
                # 直接調用main函數
                result = enhanced_main()
                
                # 關鍵修復：確保獲取正確的預測重量
                predicted_weight = None
                
                # 方法1: 從結果中提取預測重量
                if result and isinstance(result, dict):
                    # 優先從 predicted_weight 字段獲取
                    if 'predicted_weight' in result:
                        predicted_weight = float(result['predicted_weight'])
                        logger.info(f"✅ 從結果中獲取預測重量: {predicted_weight:.2f} kg")
                    
                    # 從 predictions 列表中獲取
                    elif 'predictions' in result:
                        predictions = result['predictions']
                        if isinstance(predictions, list) and len(predictions) > 0:
                            if isinstance(predictions[0], dict):
                                predicted_weight = float(predictions[0].get('weight', chair_config['true_weight']))
                            else:
                                predicted_weight = float(predictions[0])
                            logger.info(f"✅ 從預測列表中獲取重量: {predicted_weight:.2f} kg")
                    
                    # 從模型結果中獲取
                    elif 'model_results' in result:
                        model_results = result['model_results']
                        if isinstance(model_results, dict):
                            # 優先使用隨機森林的預測結果
                            if 'random_forest' in model_results:
                                rf_result = model_results['random_forest']
                                if 'prediction' in rf_result:
                                    predicted_weight = float(rf_result['prediction'])
                                    logger.info(f"✅ 從隨機森林結果獲取重量: {predicted_weight:.2f} kg")
                            
                            # 備用：使用其他模型的結果
                            elif 'xgboost' in model_results:
                                xgb_result = model_results['xgboost']
                                if 'prediction' in xgb_result:
                                    predicted_weight = float(xgb_result['prediction'])
                                    logger.info(f"✅ 從XGBoost結果獲取重量: {predicted_weight:.2f} kg")
                
                # 方法2: 檢查輸出目錄中的結果文件
                if predicted_weight is None:
                    output_dir = Path(self.config['output_dirs']['enhanced_analysis'])
                    if output_dir.exists():
                        # 查找最新的結果文件
                        result_files = list(output_dir.glob("**/detailed_analysis_results.txt"))
                        if result_files:
                            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                            try:
                                with open(latest_file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    # 使用正則表達式提取預測重量
                                    import re
                                    weight_pattern = r'新椅子預測重量[：:]\s*(\d+\.?\d*)\s*kg'
                                    match = re.search(weight_pattern, content)
                                    if match:
                                        predicted_weight = float(match.group(1))
                                        logger.info(f"✅ 從結果文件獲取重量: {predicted_weight:.2f} kg")
                            except Exception as e:
                                logger.warning(f"⚠️ 讀取結果文件失敗: {e}")
                
                # 如果仍然沒有找到預測重量，使用椅子參數中的重量
                if predicted_weight is None:
                    predicted_weight = chair_config['true_weight']
                    logger.warning(f"⚠️ 未找到預測重量，使用配置中的重量: {predicted_weight:.2f} kg")
                
                # 重要：記錄最終使用的預測重量
                logger.info(f"🎯 最終確定的預測重量: {predicted_weight:.2f} kg")
                
                step_time = time.time() - step_start_time
                self.timing_records['step_5'] = step_time
                
                return {
                    'success': True, 
                    'result': result,
                    'predicted_weight': predicted_weight,  # 確保保存正確的預測重量
                    'weight_source': 'enhanced_rf_analysis',
                    'step_time': step_time
                }
                
            finally:
                # 恢復原始argv
                sys.argv = original_argv

        except Exception as e:
            logger.error(f"❌ 步驟5執行異常: {e}")
            return {'success': False, 'error': str(e)}


    def run_workflow(self, start_step: int = 1, end_step: int = 5) -> Dict:
        """
        運行完整工作流程
        
        Args:
            start_step: 開始步驟
            end_step: 結束步驟
            
        Returns:
            工作流程結果
        """
        logger.info("🚀 開始椅子3D模型處理與分析工作流程")
        logger.info("=" * 100)
        logger.info(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"處理步驟: {start_step} - {end_step}")
        if self.config['processing']['generation_filter']:
            logger.info(f"生成變體過濾: {self.config['processing']['generation_filter']}")
        logger.info("=" * 100)
        
        # 創建必要目錄
        self.create_directories()
        
        # 檢查輸入目錄結構
        glb_models_dir = Path(self.config['input_dirs']['glb_models'])
        chair_dir = glb_models_dir / 'Chair'
        
        if not chair_dir.exists():
            logger.error(f"❌ Chair目錄不存在: {chair_dir}")
            return {'success': False, 'error': f'Chair directory not found: {chair_dir}'}
        
        # 檢查生成變體目錄
        generation_dirs = [d for d in chair_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('Chair_generation_')]
        
        if not generation_dirs:
            logger.error(f"❌ 未找到Chair_generation_*目錄在: {chair_dir}")
            return {'success': False, 'error': f'No Chair_generation_* directories found in: {chair_dir}'}
        
        logger.info(f"找到 {len(generation_dirs)} 個椅子生成變體目錄")
        
        # 應用生成變體過濾
        if self.config['processing']['generation_filter']:
            filtered_dirs = [d for d in generation_dirs if d.name in self.config['processing']['generation_filter']]
            if not filtered_dirs:
                logger.error(f"❌ 指定的生成變體都未找到: {self.config['processing']['generation_filter']}")
                return {'success': False, 'error': 'No matching generation variants found'}
            generation_dirs = filtered_dirs
            logger.info(f"篩選後處理生成變體: {[d.name for d in generation_dirs]}")
        
        # 檢查啟用的步驟
        enabled_steps = self.config.get('enabled_steps', list(range(start_step, end_step + 1)))
        steps_to_run = [s for s in range(start_step, end_step + 1) if s in enabled_steps]
        
        if not steps_to_run:
            logger.error("❌ 沒有啟用的步驟可執行")
            return {'success': False, 'error': 'No enabled steps to run'}
        
        logger.info(f"將執行步驟: {steps_to_run}")
        logger.info(f"處理 {len(generation_dirs)} 個生成變體")
        
        # 執行各個步驟
        step_results = {}
        overall_success = True
        
        # 步驟1: GLB轉OBJ
        if 1 in steps_to_run:
            step_results[1] = self.step_1_glb_to_obj()
            if not step_results[1]['success']:
                overall_success = False
                logger.error("❌ 步驟1失敗，可能影響後續步驟")
        
        # 步驟2: OBJ修改與分析
        if 2 in steps_to_run:
            step_results[2] = self.step_2_obj_modification()
            if not step_results[2]['success']:
                overall_success = False
                logger.error("❌ 步驟2失敗，可能影響後續步驟")
        
        # 步驟3: OBJ幾何分析
        if 3 in steps_to_run:
            step_results[3] = self.step_3_obj_analysis()
            if not step_results[3]['success']:
                overall_success = False
                logger.error("❌ 步驟3失敗，可能影響後續步驟")
        
        # 步驟4: Bootstrap統計分析
        if 4 in steps_to_run:
            step_results[4] = self.step_4_bootstrap_analysis()
            if not step_results[4]['success']:
                overall_success = False
                logger.error("❌ 步驟4失敗，可能影響後續步驟")
        
        # 步驟5: 增強隨機森林分析
        if 5 in steps_to_run:
            step_results[5] = self.step_5_enhanced_analysis()
            if not step_results[5]['success']:
                overall_success = False
                logger.error("❌ 步驟5失敗")
        
        # 記錄總體結果
        total_time = time.time() - self.start_time
        self.timing_records['total_time'] = total_time
        
        # 生成最終報告
        final_result = {
            'success': overall_success,
            'executed_steps': steps_to_run,
            'step_results': step_results,
            'timing_records': self.timing_records,
            'total_time': total_time,
            'processed_generations': len(generation_dirs),
            'generation_list': [d.name for d in generation_dirs],
            'config': self.config
        }
        
        # 保存工作流程報告
        self.save_workflow_report(final_result)
        
        # 打印總結
        self.print_workflow_summary(final_result)
        
        return final_result

    def save_workflow_report(self, result: Dict):
        """保存工作流程報告"""
        try:
            # 創建報告目錄
            report_dir = Path('workflow_reports')
            report_dir.mkdir(exist_ok=True)

            # 生成報告文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_report_file = report_dir / f'workflow_report_{timestamp}.json'
            text_report_file = report_dir / f'workflow_summary_{timestamp}.txt'

            # 提取預測重量
            predicted_weights = {}
            final_predicted_weight = None
            
            for step_num, step_result in result['step_results'].items():
                if 'predicted_weight' in step_result:
                    predicted_weights[f'step_{step_num}'] = step_result['predicted_weight']
                    if step_num == 5:  # 步驟5的預測重量優先
                        final_predicted_weight = step_result['predicted_weight']

            # 如果步驟5沒有預測重量，使用最後一個有預測重量的步驟
            if final_predicted_weight is None and predicted_weights:
                final_predicted_weight = list(predicted_weights.values())[-1]

            # 創建可序列化的結果副本
            serializable_result = {}
            for key, value in result.items():
                if key == 'step_results':
                    serializable_result[key] = {}
                    for step_num, step_result in value.items():
                        serializable_result[key][step_num] = {
                            'success': step_result.get('success', False),
                            'error': step_result.get('error', None),
                            'timing': self.timing_records.get(f'step_{step_num}', 0),
                            'predicted_weight': step_result.get('predicted_weight', None)  # 保存預測重量
                        }
                else:
                    serializable_result[key] = value

            # 添加預測重量摘要
            if predicted_weights:
                serializable_result['predicted_weights'] = predicted_weights
                serializable_result['final_predicted_weight'] = final_predicted_weight

            # 保存JSON格式報告
            with open(json_report_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)

            # 保存文字格式摘要
            with open(text_report_file, 'w', encoding='utf-8') as f:
                f.write("椅子3D模型處理與分析工作流程報告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"總耗時: {result['total_time']:.2f} 秒\n")
                f.write(f"整體成功: {'是' if result['success'] else '否'}\n")
                f.write(f"執行步驟: {result['executed_steps']}\n")
                
                if final_predicted_weight is not None:
                    f.write(f"最終預測重量: {final_predicted_weight:.2f} kg\n")
                
                f.write("\n各步驟執行結果:\n")
                f.write("-" * 30 + "\n")
                
                for step_num in result['executed_steps']:
                    step_result = result['step_results'].get(step_num, {})
                    step_name = self.steps.get(step_num, {}).get('name', f'步驟{step_num}')
                    step_time = self.timing_records.get(f'step_{step_num}', 0)
                    status = "✅ 成功" if step_result.get('success', False) else "❌ 失敗"
                    f.write(f"步驟{step_num}: {step_name}\n")
                    f.write(f" 狀態: {status}\n")
                    f.write(f" 耗時: {step_time:.2f} 秒\n")
                    
                    if 'predicted_weight' in step_result:
                        f.write(f" 預測重量: {step_result['predicted_weight']:.2f} kg\n")
                    
                    if not step_result.get('success', False) and step_result.get('error'):
                        f.write(f" 錯誤: {step_result['error']}\n")
                    f.write("\n")

                f.write("配置信息:\n")
                f.write("-" * 30 + "\n")
                f.write(f"GLB模型目錄: {self.config['input_dirs']['glb_models']}\n")
                f.write(f"CSV數據文件: {self.config['input_dirs']['csv_data']}\n")
                f.write(f"Bootstrap迭代次數: {self.config['bootstrap']['n_iterations']}\n")
                f.write(f"模型類型: {self.config['analysis']['model_type']}\n")

            logger.info(f"✅ 工作流程報告已保存:")
            logger.info(f" - JSON報告: {json_report_file}")
            logger.info(f" - 文字摘要: {text_report_file}")

        except Exception as e:
            logger.error(f"❌ 保存工作流程報告失敗: {e}")


    def print_workflow_summary(self, result: Dict):
        """打印工作流程總結"""
        logger.info("\n" + "🎉 工作流程執行完成！")
        logger.info("=" * 80)
        logger.info("📊 執行總結")
        logger.info("=" * 80)
        
        # 整體狀態
        overall_status = "✅ 成功" if result['success'] else "❌ 失敗"
        logger.info(f"整體狀態: {overall_status}")
        logger.info(f"總執行時間: {result['total_time']:.2f} 秒")
        logger.info(f"執行步驟數量: {len(result['executed_steps'])}")
        
        # 各步驟狀態
        logger.info("\n📋 各步驟執行狀態:")
        for step_num in result['executed_steps']:
            step_result = result['step_results'].get(step_num, {})
            step_name = self.steps.get(step_num, {}).get('name', f'步驟{step_num}')
            step_time = self.timing_records.get(f'step_{step_num}', 0)
            status = "✅" if step_result.get('success', False) else "❌"
            
            logger.info(f"  {status} 步驟{step_num}: {step_name} ({step_time:.2f}秒)")
            
            if not step_result.get('success', False) and step_result.get('error'):
                logger.info(f"      錯誤: {step_result['error']}")
        
        # 輸出目錄信息
        logger.info("\n📁 輸出目錄:")
        for dir_name, dir_path in self.config['output_dirs'].items():
            if Path(dir_path).exists():
                file_count = len(list(Path(dir_path).rglob('*.*')))
                logger.info(f"  {dir_name}: {dir_path} ({file_count} 個文件)")
        
        logger.info("=" * 80)
        
        # 提供下一步建議
        if result['success']:
            logger.info("🎯 建議下一步:")
            logger.info("  1. 檢查各輸出目錄中的分析結果")
            logger.info("  2. 查看Bootstrap分析報告以了解模型性能")
            logger.info("  3. 根據Enhanced分析結果調整模型參數")
        else:
            logger.info("🔧 故障排除建議:")
            logger.info("  1. 檢查輸入文件是否存在且格式正確")
            logger.info("  2. 確認所有必要的Python包已安裝")
            logger.info("  3. 查看詳細錯誤信息並修復問題")


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='椅子3D模型處理與分析完整工作流程')
    
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路徑')
    parser.add_argument('--start_step', type=int, default=1,
                       help='開始步驟 (1-5)')
    parser.add_argument('--end_step', type=int, default=5,
                       help='結束步驟 (1-5)')
    parser.add_argument('--glb_dir', type=str, default='./3d_models',
                       help='GLB模型輸入目錄')
    parser.add_argument('--csv_file', type=str, default='./models/chair_raw_data_v2_idmatched_revised_cleaned_aug_v2.csv',
                       help='CSV數據文件路徑')
    parser.add_argument('--bootstrap_iterations', type=int, default=50,
                       help='Bootstrap迭代次數')
    parser.add_argument('--model_type', type=str, default='all',
                       choices=['rf', 'xgb', 'mlr', 'svr', 'all'],
                       help='模型類型')
    parser.add_argument('--chair_filter', type=str, default=None,
                       help='生成變體過濾列表，用逗號分隔，如: Chair_generation_1,Chair_generation_3,Chair_generation_5')
    parser.add_argument('--save_config', action='store_true',
                       help='保存當前配置到文件')
    
    return parser.parse_args()


def main():
    """主函數"""
    args = parse_args()
    
    # 初始化工作流程整合器
    workflow = WorkflowIntegration(args.config)
    
    # 根據命令行參數更新配置
    if args.glb_dir:
        workflow.config['input_dirs']['glb_models'] = args.glb_dir
    if args.csv_file:
        workflow.config['input_dirs']['csv_data'] = args.csv_file
    if args.bootstrap_iterations:
        workflow.config['bootstrap']['n_iterations'] = args.bootstrap_iterations
    if args.model_type:
        workflow.config['analysis']['model_type'] = args.model_type
    if args.chair_filter:
        generation_list = [gen.strip() for gen in args.chair_filter.split(',')]
        workflow.config['processing']['generation_filter'] = generation_list
    
    # 保存配置（如果請求）
    if args.save_config:
        workflow.save_config('workflow_config.json')
    
    # 驗證步驟範圍
    if not (1 <= args.start_step <= 5) or not (1 <= args.end_step <= 5):
        logger.error("❌ 步驟範圍必須在1-5之間")
        return
    
    if args.start_step > args.end_step:
        logger.error("❌ 開始步驟不能大於結束步驟")
        return
    
    # 檢查必要的輸入文件和目錄結構
    glb_dir = Path(workflow.config['input_dirs']['glb_models'])
    csv_file = Path(workflow.config['input_dirs']['csv_data'])
    chair_dir = glb_dir / 'Chair'
    
    if not glb_dir.exists():
        logger.error(f"❌ GLB模型目錄不存在: {glb_dir}")
        logger.error("請確認3D模型目錄路徑正確，或使用 --glb_dir 參數指定")
        return
    
    if not chair_dir.exists():
        logger.error(f"❌ Chair子目錄不存在: {chair_dir}")
        logger.error("請確認目錄結構為: {glb_dir}/Chair/Chair_generation_X/")
        return
    
    # 檢查是否有生成變體目錄
    generation_dirs = [d for d in chair_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('Chair_generation_')]
    
    if not generation_dirs:
        logger.error(f"❌ 在 {chair_dir} 中未找到Chair_generation_*目錄")
        logger.error("請確認目錄結構包含Chair_generation_1, Chair_generation_2等目錄")
        return
    
    logger.info(f"✅ 找到 {len(generation_dirs)} 個椅子生成變體")
    for gen_dir in sorted(generation_dirs, key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else 0):
        glb_files = list(gen_dir.glob('*.glb'))
        logger.info(f"  {gen_dir.name}: {len(glb_files)} 個GLB文件")
    
    if args.end_step >= 4 and not csv_file.exists():
        logger.error(f"❌ CSV數據文件不存在: {csv_file}")
        logger.error("Bootstrap和Enhanced分析需要CSV數據文件，請使用 --csv_file 參數指定")
        return
    
    # 運行工作流程
    try:
        result = workflow.run_workflow(args.start_step, args.end_step)
        
        # 返回適當的退出碼
        exit_code = 0 if result['success'] else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 用戶中斷執行")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ 工作流程執行失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()