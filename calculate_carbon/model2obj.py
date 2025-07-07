import argparse
import os
import trimesh
from pygltflib import GLTF2
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import time


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_glb_to_obj(input_file, output_file=None):
    """
    Convert glb, obj, ply, dae, 3mf... files to .obj format
    
    Args:
        input_file (str): Input GLB file path
        output_file (str, optional): Output OBJ file path, if not specified, use input filename with .obj suffix
    
    Returns:
        result: Function success status (success, fail)
        msg: success-output file path, fail-error message
    """
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' does not exist")
        
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}.obj"
        
        logger.info(f"Loading file: {input_file}")
        scene = trimesh.load(input_file)
        
        logger.info(f"Converting to OBJ and saving to: {output_file}")
        if isinstance(scene, trimesh.Scene):
            # If it's a scene, merge all meshes and export
            mesh = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene.geometry.values()
            ])
            mesh.export(output_file, file_type='obj')
        else:
            # If it's just a single mesh, export directly
            scene.export(output_file, file_type='obj')
        result = "success"
        msg = output_file
    except Exception as e:
        result = "fail"
        msg = str(e)

    return result, msg


def batch_convert_glb_to_obj(input_dir: str, output_dir: str, file_pattern: str = "*.glb") -> Dict:
    """
    Batch convert GLB files in directory to OBJ format
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path  
        file_pattern: File matching pattern
        
    Returns:
        Conversion result dictionary
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return {'success': False, 'error': f'Input directory not found: {input_dir}'}
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find GLB files
    glb_files = list(input_path.rglob(file_pattern))
    
    if not glb_files:
        logger.warning(f"No files matching {file_pattern} found in {input_dir}")
        return {'success': True, 'converted_files': [], 'failed_files': []}
    
    logger.info(f"Found {len(glb_files)} GLB files to convert")
    
    converted_files = []
    failed_files = []
    
    for glb_file in glb_files:
        try:
            # Calculate relative path to maintain directory structure
            relative_path = glb_file.relative_to(input_path)
            obj_relative_path = relative_path.with_suffix('.obj')
            output_file = output_path / obj_relative_path
            
            # Create output subdirectory
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Converting: {relative_path} -> {obj_relative_path}")
            
            result, msg = convert_glb_to_obj(str(glb_file), str(output_file))
            
            if result == "success":
                converted_files.append({
                    'input_file': str(glb_file),
                    'output_file': str(output_file),
                    'relative_path': str(relative_path)
                })
                logger.info(f"✅ Conversion successful: {relative_path}")
            else:
                failed_files.append({
                    'input_file': str(glb_file),
                    'error': msg,
                    'relative_path': str(relative_path)
                })
                logger.error(f"❌ Conversion failed: {relative_path} - {msg}")
                
        except Exception as e:
            failed_files.append({
                'input_file': str(glb_file),
                'error': str(e),
                'relative_path': str(glb_file.relative_to(input_path))
            })
            logger.error(f"❌ Processing failed: {glb_file.relative_to(input_path)} - {e}")
    
    # Save conversion report
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
    
    logger.info(f"Conversion completed: {len(converted_files)} successful, {len(failed_files)} failed")
    logger.info(f"Conversion report saved: {report_file}")
    
    return {
        'success': True,
        'total_files': len(glb_files),
        'converted_files': converted_files,
        'failed_files': failed_files,
        'report_file': str(report_file)
    }


def process_chair_models(base_input_dir: str, base_output_dir: str) -> Dict:
    """
    Process chair model directory structure (Chair/Chair_generation_X/)
    
    Args:
        base_input_dir: Base input directory containing chair generation variants
        base_output_dir: Base output directory
        
    Returns:
        Processing result dictionary
    """
    base_input_path = Path(base_input_dir)
    base_output_path = Path(base_output_dir)
    
    if not base_input_path.exists():
        logger.error(f"Base input directory does not exist: {base_input_dir}")
        return {'success': False, 'error': f'Base input directory not found: {base_input_dir}'}
    
    # Find Chair directory
    chair_main_dir = base_input_path / 'Chair'
    if not chair_main_dir.exists():
        logger.error(f"Chair directory does not exist: {chair_main_dir}")
        return {'success': False, 'error': f'Chair directory not found: {chair_main_dir}'}
    
    # Find Chair_generation_X directories
    generation_dirs = [d for d in chair_main_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('Chair_generation_')]
    
    if not generation_dirs:
        logger.warning(f"No Chair_generation_* directories found in {chair_main_dir}")
        return {'success': True, 'generation_results': {}}
    
    logger.info(f"Found {len(generation_dirs)} chair generation variant directories")
    
    generation_results = {}
    
    # Create Chair output directory
    chair_output_dir = base_output_path / 'Chair'
    chair_output_dir.mkdir(parents=True, exist_ok=True)
    
    for gen_dir in sorted(generation_dirs, key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else 0):
        gen_id = gen_dir.name
        logger.info(f"Processing generation variant: {gen_id}")
        
        gen_output_dir = chair_output_dir / gen_id
        
        # Convert all GLB files in this generation variant directory
        gen_result = batch_convert_glb_to_obj(
            str(gen_dir), 
            str(gen_output_dir),
            "*.glb"
        )
        
        generation_results[gen_id] = gen_result
        
        if gen_result['success']:
            logger.info(f"✅ {gen_id} processing completed: converted {len(gen_result['converted_files'])} files")
        else:
            logger.error(f"❌ {gen_id} processing failed: {gen_result.get('error', 'Unknown error')}")
    
    # Generate overall report
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
    
    logger.info(f"Overall conversion completed: converted {total_converted} files, failed {total_failed}")
    logger.info(f"Overall report saved: {overall_report_file}")
    
    return {
        'success': True,
        'processed_generations': len(generation_dirs),
        'total_converted': total_converted,
        'total_failed': total_failed,
        'generation_results': generation_results,
        'report_file': str(overall_report_file)
    }


def main():
    """Main function - supports command line and programmatic calls"""
    parser = argparse.ArgumentParser(description='GLB to OBJ batch conversion tool')
    parser.add_argument('--input_dir', type=str, default='./3d_models', 
                       help='Input directory containing chair GLB models')
    parser.add_argument('--output_dir', type=str, default='./obj_models',
                       help='OBJ file output directory')
    parser.add_argument('--single_file', type=str, default=None,
                       help='Single file conversion mode')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Single file output path')
    
    args = parser.parse_args()
    
    if args.single_file:
        # Single file conversion mode
        logger.info(f"Single file conversion mode: {args.single_file}")
        result, msg = convert_glb_to_obj(args.single_file, args.output_file)
        if result == "success":
            logger.info(f"✅ Conversion successful: {msg}")
        else:
            logger.error(f"❌ Conversion failed: {msg}")
    else:
        # Batch conversion mode
        logger.info(f"Batch conversion mode: {args.input_dir} -> {args.output_dir}")
        result = process_chair_models(args.input_dir, args.output_dir)
        
        if result['success']:
            logger.info("🎉 Batch conversion completed!")
        else:
            logger.error(f"❌ Batch conversion failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
