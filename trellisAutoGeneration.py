import os
import sys
import glob
import random
import numpy as np
from pathlib import Path
from PIL import Image
import imageio
import torch
from typing import List, Tuple, Dict
import logging
import shutil
from easydict import EasyDict as edict
import time
from datetime import datetime, timedelta
import json

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TRELLIS相關導入
try:
    # 設置環境變數
    os.environ['SPCONV_ALGO'] = 'native'
    
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.representations import Gaussian, MeshExtractResult
    from trellis.utils import render_utils, postprocessing_utils
except ImportError as e:
    logger.error(f"TRELLIS導入失敗: {e}")
    logger.error("請確保已正確安裝TRELLIS並激活conda環境")
    sys.exit(1)

class TrellisChairGenerator:
    """基於椅子檢測成功結果的TRELLIS 3D生成器"""
    
    def __init__(self, model_name="JeffreyXiang/TRELLIS-image-large"):
        """
        初始化生成器
        
        Args:
            model_name: 預訓練模型名稱
        """
        self.model_name = model_name
        self.pipeline = None
        self.supported_formats = ['.webp', '.png', '.jpg', '.jpeg']
        
        # 時間記錄
        self.timing_records = {
            'model_loading': 0,
            'preprocessing': {},  # {chair_name: time}
            'generations': {},    # {output_name: {'total': time, 'steps': {...}}}
            'chairs': {},         # {chair_name: total_time}
            'start_time': None,
            'end_time': None
        }
        
    def load_model(self):
        """載入TRELLIS模型"""
        try:
            logger.info(f"正在載入模型: {self.model_name}")
            start_time = time.time()
            
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(self.model_name)
            self.pipeline.cuda()
            
            load_time = time.time() - start_time
            self.timing_records['model_loading'] = load_time
            logger.info(f"模型載入成功 (耗時: {load_time:.2f}秒)")
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            raise
    
    def find_chair_images(self, identify_pass_dir: str) -> Dict[str, List[str]]:
        """
        在 identify_pass 資料夾中尋找各椅子的成功檢測圖片
        
        Args:
            identify_pass_dir: identify_pass 資料夾路徑
            
        Returns:
            {chair_id: [image_paths]} 的字典
        """
        chair_images = {}
        identify_pass_path = Path(identify_pass_dir)
        
        if not identify_pass_path.exists():
            logger.error(f"identify_pass 資料夾不存在: {identify_pass_path}")
            return chair_images
        
        # 遍歷所有椅子資料夾（排除 original_img）
        for chair_folder in identify_pass_path.iterdir():
            if chair_folder.is_dir() and chair_folder.name != 'original_img':
                chair_id = chair_folder.name
                
                # 尋找該椅子的所有圖片
                image_files = []
                for ext in self.supported_formats:
                    # 大小寫不敏感
                    for case_ext in [ext.lower(), ext.upper()]:
                        pattern = chair_folder / f"*{case_ext}"
                        image_files.extend(glob.glob(str(pattern)))
                
                # 去重並排序
                image_files = sorted(list(set(image_files)))
                
                if image_files:
                    chair_images[chair_id] = image_files
                    logger.info(f"找到 {chair_id}: {len(image_files)} 張成功檢測的圖片")
                else:
                    logger.warning(f"{chair_id}: 沒有找到圖片檔案")
        
        logger.info(f"總共找到 {len(chair_images)} 個椅子的成功檢測圖片")
        return chair_images
    
    def preprocess_images(self, image_paths: List[str]) -> List[Image.Image]:
        """
        使用官方的圖片預處理方法
        
        Args:
            image_paths: 圖片檔案路徑列表
            
        Returns:
            預處理後的圖片列表
        """
        processed_images = []
        
        for img_path in image_paths:
            try:
                # 載入圖片
                image = Image.open(img_path)
                logger.info(f"正在預處理: {Path(img_path).name}")
                
                # 使用官方的預處理方法
                processed_image = self.pipeline.preprocess_image(image)
                processed_images.append(processed_image)
                
            except Exception as e:
                logger.warning(f"預處理圖片 {img_path} 時發生錯誤: {e}")
                continue
                
        logger.info(f"成功預處理 {len(processed_images)} 張圖片")
        return processed_images
    
    def pack_state(self, gs: Gaussian, mesh: MeshExtractResult) -> dict:
        """
        打包狀態（參考官方app.py）
        """
        return {
            'gaussian': {
                **gs.init_params,
                '_xyz': gs._xyz.cpu().numpy(),
                '_features_dc': gs._features_dc.cpu().numpy(),
                '_scaling': gs._scaling.cpu().numpy(),
                '_rotation': gs._rotation.cpu().numpy(),
                '_opacity': gs._opacity.cpu().numpy(),
            },
            'mesh': {
                'vertices': mesh.vertices.cpu().numpy(),
                'faces': mesh.faces.cpu().numpy(),
            },
        }
    
    def unpack_state(self, state: dict) -> Tuple[Gaussian, edict]:
        """
        解包狀態（參考官方app.py）
        """
        gs = Gaussian(
            aabb=state['gaussian']['aabb'],
            sh_degree=state['gaussian']['sh_degree'],
            mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
            scaling_bias=state['gaussian']['scaling_bias'],
            opacity_bias=state['gaussian']['opacity_bias'],
            scaling_activation=state['gaussian']['scaling_activation'],
        )
        gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
        gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
        gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
        gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
        gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
        
        mesh = edict(
            vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
            faces=torch.tensor(state['mesh']['faces'], device='cuda'),
        )
        
        return gs, mesh
    
    def generate_3d_multi_image(self, images: List[Image.Image], seed: int = None,
                               ss_guidance_strength: float = 7.5,
                               ss_sampling_steps: int = 12,
                               slat_guidance_strength: float = 3.0,
                               slat_sampling_steps: int = 12,
                               multiimage_algo: str = "stochastic") -> dict:
        """
        使用多圖片生成3D模型（基於官方app.py的image_to_3d函數）
        
        Args:
            images: 預處理後的圖片列表
            seed: 隨機種子
            ss_guidance_strength: Sparse Structure階段的guidance強度
            ss_sampling_steps: Sparse Structure階段的採樣步數
            slat_guidance_strength: Structured Latent階段的guidance強度
            slat_sampling_steps: Structured Latent階段的採樣步數
            multiimage_algo: 多圖片算法 ("multidiffusion" 或 "stochastic")
            
        Returns:
            生成結果字典
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
        try:
            logger.info(f"開始多圖片3D生成，使用 {len(images)} 張圖片，種子: {seed}")
            logger.info(f"算法: {multiimage_algo}")
            
            # 使用官方的multi-image pipeline
            outputs = self.pipeline.run_multi_image(
                images,
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=False,  # 圖片已經預處理過了
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
                mode=multiimage_algo,
            )
            
            logger.info("3D模型生成成功")
            return outputs
            
        except Exception as e:
            logger.error(f"3D模型生成失敗: {e}")
            raise
    
    def render_preview_video(self, outputs: dict, output_dir: str, base_name: str):
        """
        渲染預覽影片（參考官方app.py）
        """
        try:
            # 渲染Gaussian和Mesh的影片
            video_gaussian = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
            video_mesh = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
            
            # 合併兩個影片（左右並排）
            video_combined = [np.concatenate([video_gaussian[i], video_mesh[i]], axis=1) 
                            for i in range(len(video_gaussian))]
            
            # 保存影片
            video_path = Path(output_dir) / f"{base_name}_preview.mp4"
            imageio.mimsave(str(video_path), video_combined, fps=15)
            logger.info(f"預覽影片已保存: {video_path}")
            
            return str(video_path)
            
        except Exception as e:
            logger.error(f"渲染預覽影片時發生錯誤: {e}")
            return None
    
    def extract_glb(self, outputs: dict, output_dir: str, base_name: str,
                   mesh_simplify: float = 0.95, texture_size: int = 1024) -> str:
        """
        提取GLB檔案（參考官方app.py）
        """
        try:
            gs = outputs['gaussian'][0]
            mesh = outputs['mesh'][0]
            
            # 生成GLB
            glb = postprocessing_utils.to_glb(
                gs, mesh, 
                simplify=mesh_simplify, 
                texture_size=texture_size, 
                verbose=False
            )
            
            # 保存GLB檔案
            glb_path = Path(output_dir) / f"{base_name}.glb"
            glb.export(str(glb_path))
            logger.info(f"GLB檔案已保存: {glb_path}")
            
            return str(glb_path)
            
        except Exception as e:
            logger.error(f"提取GLB時發生錯誤: {e}")
            return None
    
    def extract_gaussian(self, outputs: dict, output_dir: str, base_name: str) -> str:
        """
        提取Gaussian點雲檔案（參考官方app.py）
        """
        try:
            gs = outputs['gaussian'][0]
            
            # 保存PLY檔案
            gaussian_path = Path(output_dir) / f"{base_name}_gaussian.ply"
            gs.save_ply(str(gaussian_path))
            logger.info(f"Gaussian點雲已保存: {gaussian_path}")
            
            return str(gaussian_path)
            
        except Exception as e:
            logger.error(f"提取Gaussian時發生錯誤: {e}")
            return None
    
    def process_single_chair(self, chair_id: str, image_paths: List[str], 
                           output_base_dir: str, generations_per_chair: int = 10,
                           start_from_generation: int = None) -> bool:
        """
        處理單個椅子的所有生成
        
        Args:
            chair_id: 椅子ID（如 Chair1）
            image_paths: 該椅子的圖片路徑列表
            output_base_dir: 輸出基礎目錄
            generations_per_chair: 每個椅子的生成數量
            start_from_generation: 從指定生成開始（用於中斷後繼續）
            
        Returns:
            是否成功處理
        """
        chair_start_time = time.time()
        
        logger.info(f"開始處理 {chair_id}，找到 {len(image_paths)} 張成功檢測的圖片")
        
        # 預處理所有圖片（一次性完成）
        try:
            preprocess_start = time.time()
            processed_images = self.preprocess_images(image_paths)
            preprocess_time = time.time() - preprocess_start
            self.timing_records['preprocessing'][chair_id] = preprocess_time
            
            if not processed_images:
                logger.error(f"沒有成功預處理任何圖片: {chair_id}")
                return False
                
            logger.info(f"✅ {chair_id} 的圖片預處理完成 (耗時: {preprocess_time:.2f}秒)，將用於生成 {generations_per_chair} 個變體")
            
        except Exception as e:
            logger.error(f"預處理 {chair_id} 的圖片時發生錯誤: {e}")
            return False
        
        # 創建椅子的輸出目錄
        chair_output_dir = Path(output_base_dir) / chair_id
        chair_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成多個變體
        success_count = 0
        for gen_idx in range(1, generations_per_chair + 1):
            # 檢查是否需要跳過（用於中斷後繼續）
            if start_from_generation is not None and gen_idx < start_from_generation:
                continue
                
            output_name = f"{chair_id}_generation_{gen_idx}"
            output_folder = chair_output_dir / output_name
            
            # 檢查是否已經生成過
            if output_folder.exists():
                existing_glb = list(output_folder.glob("*.glb"))
                if existing_glb:
                    logger.info(f"跳過 {output_name}：已存在GLB檔案")
                    success_count += 1
                    continue
            
            # 創建輸出資料夾
            output_folder.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"正在生成 {output_name} ({gen_idx}/{generations_per_chair})")
            
            # 記錄單個生成的時間
            gen_start_time = time.time()
            step_times = {}
            
            try:
                # 生成唯一的種子
                seed = hash(f"{chair_id}_{gen_idx}") % (2**31)
                
                # 生成3D模型
                step_start = time.time()
                outputs = self.generate_3d_multi_image(
                    processed_images, 
                    seed=seed,
                    multiimage_algo="stochastic"  # 可以改為 "multidiffusion"
                )
                step_times['3d_generation'] = time.time() - step_start
                
                # 保存各種格式
                # 1. 渲染預覽影片
                step_start = time.time()
                self.render_preview_video(outputs, output_folder, output_name)
                step_times['video_rendering'] = time.time() - step_start
                
                # 2. 提取GLB檔案
                step_start = time.time()
                self.extract_glb(outputs, output_folder, output_name)
                step_times['glb_extraction'] = time.time() - step_start
                
                # 3. 提取Gaussian點雲
                step_start = time.time()
                self.extract_gaussian(outputs, output_folder, output_name)
                step_times['gaussian_extraction'] = time.time() - step_start
                
                # 4. 保存狀態（可選，用於後續處理）
                step_start = time.time()
                state = self.pack_state(outputs['gaussian'][0], outputs['mesh'][0])
                state_path = output_folder / f"{output_name}_state.pt"
                torch.save(state, str(state_path))
                step_times['state_saving'] = time.time() - step_start
                
                # 5. 保存生成元數據
                metadata = {
                    'chair_id': chair_id,
                    'generation_index': gen_idx,
                    'seed': seed,
                    'source_images': [Path(p).name for p in image_paths],
                    'generation_time': time.time() - gen_start_time,
                    'generated_at': datetime.now().isoformat()
                }
                metadata_path = output_folder / f"{output_name}_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                # 清理GPU記憶體
                torch.cuda.empty_cache()
                
                # 記錄總時間
                gen_total_time = time.time() - gen_start_time
                self.timing_records['generations'][output_name] = {
                    'total': gen_total_time,
                    'steps': step_times
                }
                
                logger.info(f"✅ 完成生成 {output_name} (總耗時: {gen_total_time:.2f}秒)")
                success_count += 1
                
            except Exception as e:
                logger.error(f"生成 {output_name} 時發生錯誤: {e}")
                continue
                
        # 記錄椅子總時間
        chair_total_time = time.time() - chair_start_time
        self.timing_records['chairs'][chair_id] = chair_total_time
        
        logger.info(f"完成 {chair_id}：成功生成 {success_count}/{generations_per_chair} 個變體 (總耗時: {chair_total_time:.2f}秒)")
        return success_count > 0
    
    def save_timing_report(self, output_path: str = "timing_report.json"):
        """保存時間報告到JSON檔案"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.timing_records, f, indent=2, ensure_ascii=False)
            logger.info(f"時間報告已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存時間報告失敗: {e}")
    
    def print_timing_summary(self):
        """打印時間統計摘要"""
        logger.info("\n" + "=" * 80)
        logger.info("⏱️  時間統計報告")
        logger.info("=" * 80)
        
        # 總體時間
        if self.timing_records['start_time'] and self.timing_records['end_time']:
            total_duration = self.timing_records['end_time'] - self.timing_records['start_time']
            logger.info(f"📊 總執行時間: {str(timedelta(seconds=int(total_duration)))}")
            logger.info(f"🚀 模型載入時間: {self.timing_records['model_loading']:.2f}秒")
        
        # 預處理時間統計
        if self.timing_records['preprocessing']:
            preprocess_times = list(self.timing_records['preprocessing'].values())
            logger.info(f"\n📸 圖片預處理統計:")
            logger.info(f"   - 平均時間: {np.mean(preprocess_times):.2f}秒")
            logger.info(f"   - 最快: {min(preprocess_times):.2f}秒")
            logger.info(f"   - 最慢: {max(preprocess_times):.2f}秒")
        
        # 生成時間統計
        if self.timing_records['generations']:
            gen_times = [v['total'] for v in self.timing_records['generations'].values()]
            logger.info(f"\n🎨 3D生成統計:")
            logger.info(f"   - 生成數量: {len(gen_times)}")
            logger.info(f"   - 平均時間: {np.mean(gen_times):.2f}秒")
            logger.info(f"   - 最快: {min(gen_times):.2f}秒")
            logger.info(f"   - 最慢: {max(gen_times):.2f}秒")
            logger.info(f"   - 總生成時間: {sum(gen_times):.2f}秒")
            
            # 各步驟平均時間
            step_names = ['3d_generation', 'video_rendering', 'glb_extraction', 
                         'gaussian_extraction', 'state_saving']
            step_times_avg = {}
            
            for step in step_names:
                times = [v['steps'].get(step, 0) for v in self.timing_records['generations'].values() 
                        if step in v['steps']]
                if times:
                    step_times_avg[step] = np.mean(times)
            
            if step_times_avg:
                logger.info(f"\n📈 各步驟平均時間:")
                logger.info(f"   - 3D生成: {step_times_avg.get('3d_generation', 0):.2f}秒")
                logger.info(f"   - 影片渲染: {step_times_avg.get('video_rendering', 0):.2f}秒")
                logger.info(f"   - GLB提取: {step_times_avg.get('glb_extraction', 0):.2f}秒")
                logger.info(f"   - Gaussian提取: {step_times_avg.get('gaussian_extraction', 0):.2f}秒")
                logger.info(f"   - 狀態保存: {step_times_avg.get('state_saving', 0):.2f}秒")
        
        # 各椅子時間
        if self.timing_records['chairs']:
            logger.info(f"\n🪑 各椅子處理時間:")
            for chair, time_spent in sorted(self.timing_records['chairs'].items()):
                logger.info(f"   - {chair}: {time_spent:.2f}秒 ({str(timedelta(seconds=int(time_spent)))})")
        
        logger.info("=" * 80)
    
    def run_batch_generation(self, identify_pass_dir: str = "./image_identify/identify_pass",
                           output_dir: str = "chair_3d_models",
                           chair_filter: List[str] = None,
                           generations_per_chair: int = 10,
                           start_from_chair: str = None,
                           start_from_generation: int = None):
        """
        批次生成主函數 - 基於椅子檢測成功結果
        
        Args:
            identify_pass_dir: identify_pass 資料夾路徑
            output_dir: 3D模型輸出目錄
            chair_filter: 指定要處理的椅子列表，如 ['Chair1', 'Chair5']，None表示處理所有
            generations_per_chair: 每個椅子生成的變體數量
            start_from_chair: 從指定椅子開始（用於中斷後繼續）
            start_from_generation: 從指定生成開始（用於中斷後繼續）
        """
        # 記錄開始時間
        self.timing_records['start_time'] = time.time()
        start_datetime = datetime.now()
        
        if self.pipeline is None:
            self.load_model()
        
        # 尋找所有椅子的成功檢測圖片
        chair_images = self.find_chair_images(identify_pass_dir)
        
        if not chair_images:
            logger.error("未找到任何椅子的成功檢測圖片")
            return
        
        # 應用椅子篩選
        if chair_filter:
            filtered_chair_images = {k: v for k, v in chair_images.items() if k in chair_filter}
            if not filtered_chair_images:
                logger.error(f"指定的椅子 {chair_filter} 都沒有找到成功檢測圖片")
                return
            chair_images = filtered_chair_images
            logger.info(f"篩選後處理椅子: {list(chair_images.keys())}")
        
        total_chairs = len(chair_images)
        total_generations = total_chairs * generations_per_chair
        
        # 創建輸出目錄
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("🚀 開始TRELLIS椅子3D生成")
        logger.info("=" * 60)
        logger.info(f"開始時間: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"成功檢測圖片來源: {identify_pass_dir}")
        logger.info(f"3D模型輸出目錄: {output_dir}")
        logger.info(f"找到椅子: {list(chair_images.keys())}")
        logger.info(f"每個椅子生成: {generations_per_chair} 個變體")
        logger.info(f"總共將生成: {total_generations} 個3D模型")
        logger.info(f"模型: {self.model_name}")
        logger.info("=" * 60)
        
        successful_chairs = 0
        total_successful_generations = 0
        
        for chair_id, image_paths in sorted(chair_images.items()):
            # 檢查是否需要跳過（用於中斷後繼續）
            if start_from_chair is not None and chair_id != start_from_chair:
                if chair_id < start_from_chair:  # 假設椅子ID可比較
                    continue
                else:
                    start_from_chair = None  # 找到目標椅子後重置
            
            # 決定從哪個生成開始（只在指定的椅子上生效）
            generation_start = None
            if start_from_chair is not None and chair_id == start_from_chair:
                generation_start = start_from_generation
            
            # 處理單個椅子
            success = self.process_single_chair(
                chair_id, 
                image_paths,
                output_dir,
                generations_per_chair,
                generation_start
            )
            
            if success:
                successful_chairs += 1
                
                # 統計成功的生成數量
                chair_output_dir = Path(output_dir) / chair_id
                if chair_output_dir.exists():
                    generation_folders = list(chair_output_dir.glob(f"{chair_id}_generation_*"))
                    chair_successes = sum(1 for folder in generation_folders 
                                        if list(folder.glob("*.glb")))
                    total_successful_generations += chair_successes
        
        # 記錄結束時間
        self.timing_records['end_time'] = time.time()
        end_datetime = datetime.now()
        
        logger.info("=" * 60)
        logger.info("🎉 批次生成完成！")
        logger.info("=" * 60)
        logger.info(f"結束時間: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"成功處理椅子: {successful_chairs}/{total_chairs}")
        logger.info(f"成功生成3D模型: {total_successful_generations}/{total_generations}")
        logger.info("=" * 60)
        
        # 打印時間統計
        self.print_timing_summary()
        
        # 保存時間報告
        report_path = Path(output_dir) / f"timing_report_{start_datetime.strftime('%Y%m%d_%H%M%S')}.json"
        self.save_timing_report(str(report_path))
        
        # 生成處理摘要報告
        self.generate_summary_report(output_dir, chair_images, start_datetime)
    
    def generate_summary_report(self, output_dir: str, chair_images: Dict[str, List[str]], 
                               start_datetime: datetime):
        """生成處理摘要報告"""
        report_path = Path(output_dir) / f"generation_summary_{start_datetime.strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("TRELLIS 椅子3D生成摘要報告\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"生成時間: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型: {self.model_name}\n\n")
                
                f.write("椅子處理統計:\n")
                f.write("-" * 40 + "\n")
                
                for chair_id, image_paths in sorted(chair_images.items()):
                    f.write(f"{chair_id}:\n")
                    f.write(f"  成功檢測圖片數量: {len(image_paths)}\n")
                    
                    # 檢查生成結果
                    chair_output_dir = Path(output_dir) / chair_id
                    if chair_output_dir.exists():
                        generation_folders = list(chair_output_dir.glob(f"{chair_id}_generation_*"))
                        successful_gens = sum(1 for folder in generation_folders 
                                            if list(folder.glob("*.glb")))
                        f.write(f"  成功生成3D模型: {successful_gens}\n")
                    else:
                        f.write(f"  成功生成3D模型: 0\n")
                    
                    # 處理時間
                    if chair_id in self.timing_records['chairs']:
                        chair_time = self.timing_records['chairs'][chair_id]
                        f.write(f"  處理時間: {chair_time:.2f}秒\n")
                    
                    f.write(f"  來源圖片:\n")
                    for img_path in image_paths:
                        f.write(f"    - {Path(img_path).name}\n")
                    f.write("\n")
                
                # 整體統計
                f.write("整體統計:\n")
                f.write("-" * 40 + "\n")
                f.write(f"處理椅子數量: {len(chair_images)}\n")
                
                if self.timing_records['start_time'] and self.timing_records['end_time']:
                    total_duration = self.timing_records['end_time'] - self.timing_records['start_time']
                    f.write(f"總執行時間: {str(timedelta(seconds=int(total_duration)))}\n")
                
                if self.timing_records['generations']:
                    total_generations = len(self.timing_records['generations'])
                    f.write(f"總生成數量: {total_generations}\n")
                    
                    gen_times = [v['total'] for v in self.timing_records['generations'].values()]
                    f.write(f"平均生成時間: {np.mean(gen_times):.2f}秒\n")
            
            logger.info(f"處理摘要報告已保存: {report_path}")
            
        except Exception as e:
            logger.error(f"生成摘要報告失敗: {e}")

def main():
    """主函數"""
    # 🔧 配置參數
    config = {
        "identify_pass_dir": "./image_identify/identify_pass",  # 成功檢測圖片的路徑
        "output_dir": "3d_models",  # 3D模型輸出目錄
        "generations_per_chair": 10,  # 每個椅子生成的變體數量
        "model_name": "JeffreyXiang/TRELLIS-image-large",
        
        # 🎯 椅子篩選（可選）
        "chair_filter": None,  # 例如：["Chair1", "Chair3", "Chair5"] 或 None（處理所有）
        
        # 🔄 中斷後繼續的參數（可選）
        "start_from_chair": None,  # 例如："Chair5"（從Chair5開始）
        "start_from_generation": None,  # 例如：3（從第3個生成開始）
    }
    
    # 檢查系統環境
    if not torch.cuda.is_available():
        logger.error("❌ 未檢測到CUDA，請確保有可用的GPU")
        sys.exit(1)
        
    logger.info(f"🔥 使用GPU: {torch.cuda.get_device_name()}")
    logger.info(f"💾 GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 檢查成功檢測目錄
    identify_pass_dir = Path(config["identify_pass_dir"])
    if not identify_pass_dir.exists():
        logger.error(f"❌ identify_pass 目錄不存在: {identify_pass_dir}")
        logger.error("請先運行椅子檢測程式生成成功檢測結果")
        sys.exit(1)
    
    # 創建生成器並運行
    generator = TrellisChairGenerator(config["model_name"])
    
    try:
        generator.run_batch_generation(
            identify_pass_dir=config["identify_pass_dir"],
            output_dir=config["output_dir"],
            chair_filter=config["chair_filter"],
            generations_per_chair=config["generations_per_chair"],
            start_from_chair=config["start_from_chair"],
            start_from_generation=config["start_from_generation"]
        )
    except KeyboardInterrupt:
        logger.info("\n🛑 收到中斷信號，正在安全退出...")
        # 仍然保存當前的時間記錄
        generator.timing_records['end_time'] = time.time()
        generator.print_timing_summary()
        report_path = Path(config["output_dir"]) / f"timing_report_interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_timing_report(str(report_path))
    except Exception as e:
        logger.error(f"❌ 批次生成失敗: {e}")
        raise

if __name__ == "__main__":
    main()