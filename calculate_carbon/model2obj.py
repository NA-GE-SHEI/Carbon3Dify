import argparse
import os
import trimesh
from pygltflib import GLTF2
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import time

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_glb_to_obj(input_file, output_file=None):
    """
    將 glb, obj, ply, dae, 3mf... 文件轉換為 .obj 格式
    
    參數:
        input_file (str): 輸入的 GLB 文件路徑
        output_file (str, optional): 輸出的 OBJ 文件路徑，若未指定則使用輸入文件名但改為 .obj 後綴
    
    返回:
        result: function成功與否(success, fail)
        msg: success-輸出文件的路徑, fail-錯誤信息
    """
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"輸入文件 '{input_file}' 不存在")
        
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}.obj"
        
        logger.info(f"加載文件: {input_file}")
        scene = trimesh.load(input_file)
        
        logger.info(f"轉換為 OBJ 並保存至: {output_file}")
        if isinstance(scene, trimesh.Scene):
            # 如果是場景，合併所有網格並導出
            mesh = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene.geometry.values()
            ])
            mesh.export(output_file, file_type='obj')
        else:
            # 如果只是單個網格，直接導出
            scene.export(output_file, file_type='obj')
        result = "success"
        msg = output_file
    except Exception as e:
        result = "fail"
        msg = str(e)

    return result, msg

def batch_convert_glb_to_obj(input_dir: str, output_dir: str, file_pattern: str = "*.glb") -> Dict:
    """
    批量轉換目錄中的GLB文件為OBJ格式
    
    Args:
        input_dir: 輸入目錄路徑
        output_dir: 輸出目錄路徑  
        file_pattern: 文件匹配模式
        
    Returns:
        轉換結果字典
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"輸入目錄不存在: {input_dir}")
        return {'success': False, 'error': f'Input directory not found: {input_dir}'}
    
    # 創建輸出目錄
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找GLB文件
    glb_files = list(input_path.rglob(file_pattern))
    
    if not glb_files:
        logger.warning(f"在 {input_dir} 中未找到匹配 {file_pattern} 的文件")
        return {'success': True, 'converted_files': [], 'failed_files': []}
    
    logger.info(f"找到 {len(glb_files)} 個GLB文件待轉換")
    
    converted_files = []
    failed_files = []
    
    for glb_file in glb_files:
        try:
            # 計算相對路徑以保持目錄結構
            relative_path = glb_file.relative_to(input_path)
            obj_relative_path = relative_path.with_suffix('.obj')
            output_file = output_path / obj_relative_path
            
            # 創建輸出子目錄
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"轉換: {relative_path} -> {obj_relative_path}")
            
            result, msg = convert_glb_to_obj(str(glb_file), str(output_file))
            
            if result == "success":
                converted_files.append({
                    'input_file': str(glb_file),
                    'output_file': str(output_file),
                    'relative_path': str(relative_path)
                })
                logger.info(f"✅ 轉換成功: {relative_path}")
            else:
                failed_files.append({
                    'input_file': str(glb_file),
                    'error': msg,
                    'relative_path': str(relative_path)
                })
                logger.error(f"❌ 轉換失敗: {relative_path} - {msg}")
                
        except Exception as e:
            failed_files.append({
                'input_file': str(glb_file),
                'error': str(e),
                'relative_path': str(glb_file.relative_to(input_path))
            })
            logger.error(f"❌ 處理失敗: {glb_file.relative_to(input_path)} - {e}")
    
    # 保存轉換報告
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_directory': str(input_path),
        'output_directory': str(output_path),
        'total_files': len(glb_files),
        'successful_conversions': len(converted_files),
        'failed_conversions': len(failed_files),
        'converted_files': converted_files,
        'failed_files': failed_files
    }
    
    report_file = output_path / 'conversion_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"轉換完成: 成功 {len(converted_files)} 個，失敗 {len(failed_files)} 個")
    logger.info(f"轉換報告已保存: {report_file}")
    
    return {
        'success': True,
        'total_files': len(glb_files),
        'converted_files': converted_files,
        'failed_files': failed_files,
        'report_file': str(report_file)
    }

def process_chair_models(base_input_dir: str, base_output_dir: str) -> Dict:
    """
    處理椅子模型目錄結構 (Chair/Chair_generation_X/)
    
    Args:
        base_input_dir: 包含椅子生成變體的基礎輸入目錄
        base_output_dir: 基礎輸出目錄
        
    Returns:
        處理結果字典
    """
    base_input_path = Path(base_input_dir)
    base_output_path = Path(base_output_dir)
    
    if not base_input_path.exists():
        logger.error(f"基礎輸入目錄不存在: {base_input_dir}")
        return {'success': False, 'error': f'Base input directory not found: {base_input_dir}'}
    
    # 查找Chair目錄
    chair_main_dir = base_input_path / 'Chair'
    if not chair_main_dir.exists():
        logger.error(f"Chair目錄不存在: {chair_main_dir}")
        return {'success': False, 'error': f'Chair directory not found: {chair_main_dir}'}
    
    # 查找Chair_generation_X目錄
    generation_dirs = [d for d in chair_main_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('Chair_generation_')]
    
    if not generation_dirs:
        logger.warning(f"在 {chair_main_dir} 中未找到Chair_generation_*目錄")
        return {'success': True, 'generation_results': {}}
    
    logger.info(f"找到 {len(generation_dirs)} 個椅子生成變體目錄")
    
    generation_results = {}
    
    # 創建Chair輸出目錄
    chair_output_dir = base_output_path / 'Chair'
    chair_output_dir.mkdir(parents=True, exist_ok=True)
    
    for gen_dir in sorted(generation_dirs, key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else 0):
        gen_id = gen_dir.name
        logger.info(f"處理生成變體: {gen_id}")
        
        gen_output_dir = chair_output_dir / gen_id
        
        # 轉換該生成變體目錄中的所有GLB文件
        gen_result = batch_convert_glb_to_obj(
            str(gen_dir), 
            str(gen_output_dir),
            "*.glb"
        )
        
        generation_results[gen_id] = gen_result
        
        if gen_result['success']:
            logger.info(f"✅ {gen_id} 處理完成: 轉換了 {len(gen_result['converted_files'])} 個文件")
        else:
            logger.error(f"❌ {gen_id} 處理失敗: {gen_result.get('error', 'Unknown error')}")
    
    # 生成總體報告
    total_converted = sum(len(result.get('converted_files', [])) for result in generation_results.values())
    total_failed = sum(len(result.get('failed_files', [])) for result in generation_results.values())
    
    overall_report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'base_input_directory': str(base_input_path),
        'base_output_directory': str(base_output_path),
        'chair_main_directory': str(chair_main_dir),
        'processed_generations': len(generation_dirs),
        'total_converted_files': total_converted,
        'total_failed_files': total_failed,
        'generation_results': generation_results
    }
    
    overall_report_file = base_output_path / 'overall_conversion_report.json'
    base_output_path.mkdir(parents=True, exist_ok=True)
    with open(overall_report_file, 'w', encoding='utf-8') as f:
        json.dump(overall_report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"總體轉換完成: 轉換了 {total_converted} 個文件，失敗 {total_failed} 個")
    logger.info(f"總體報告已保存: {overall_report_file}")
    
    return {
        'success': True,
        'processed_generations': len(generation_dirs),
        'total_converted': total_converted,
        'total_failed': total_failed,
        'generation_results': generation_results,
        'report_file': str(overall_report_file)
    }

def main():
    """主函數 - 支持命令行調用和程序化調用"""
    parser = argparse.ArgumentParser(description='GLB to OBJ 批量轉換工具')
    parser.add_argument('--input_dir', type=str, default='./3d_models', 
                       help='包含椅子GLB模型的輸入目錄')
    parser.add_argument('--output_dir', type=str, default='./obj_models',
                       help='OBJ文件輸出目錄')
    parser.add_argument('--single_file', type=str, default=None,
                       help='單個文件轉換模式')
    parser.add_argument('--output_file', type=str, default=None,
                       help='單個文件輸出路徑')
    
    args = parser.parse_args()
    
    if args.single_file:
        # 單文件轉換模式
        logger.info(f"單文件轉換模式: {args.single_file}")
        result, msg = convert_glb_to_obj(args.single_file, args.output_file)
        if result == "success":
            logger.info(f"✅ 轉換成功: {msg}")
        else:
            logger.error(f"❌ 轉換失敗: {msg}")
    else:
        # 批量轉換模式
        logger.info(f"批量轉換模式: {args.input_dir} -> {args.output_dir}")
        result = process_chair_models(args.input_dir, args.output_dir)
        
        if result['success']:
            logger.info("🎉 批量轉換完成！")
        else:
            logger.error(f"❌ 批量轉換失敗: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()