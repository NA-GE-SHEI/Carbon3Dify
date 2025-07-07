import numpy as np
from collections import defaultdict, Counter
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import csv

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeometryAnalyzer:
    def __init__(self):
        self.analysis_results = {}
        
    def load_obj(self, file_path):
        """讀取 OBJ 檔案並返回頂點和麵片"""
        vertices = []
        faces = []
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    vertices.append(vertex)
                elif line.startswith('f '):
                    parts = line.strip().split()
                    face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    faces.append(face)
        
        return np.array(vertices), np.array(faces)

    def build_connectivity_graph(self, faces):
        """建立頂點之間的連通性圖"""
        graph = defaultdict(set)
        
        for face in faces:
            for i in range(len(face)):
                v1 = face[i]
                v2 = face[(i + 1) % len(face)]
                graph[v1].add(v2)
                graph[v2].add(v1)
        
        return graph

    def find_connected_components(self, graph, num_vertices):
        """使用迭代 DFS 找到所有連通分量"""
        visited = set()
        components = []
        
        for start in range(num_vertices):
            if start not in visited:
                component = set()
                stack = [start]
                
                while stack:
                    vertex = stack.pop()
                    if vertex not in visited:
                        visited.add(vertex)
                        component.add(vertex)
                        stack.extend(n for n in graph[vertex] if n not in visited)
                
                components.append(component)
        
        return components

    def analyze_edge_manifold(self, faces):
        """分析邊的流形性質"""
        edge_count = defaultdict(int)
        
        for face in faces:
            for i in range(len(face)):
                v1, v2 = face[i], face[(i + 1) % len(face)]
                edge = tuple(sorted([v1, v2]))
                edge_count[edge] += 1
        
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        non_manifold_edges = [edge for edge, count in edge_count.items() if count > 2]
        manifold_edges = [edge for edge, count in edge_count.items() if count == 2]
        
        return {
            'boundary_edges': len(boundary_edges),
            'non_manifold_edges': len(non_manifold_edges),
            'manifold_edges': len(manifold_edges),
            'total_edges': len(edge_count),
            'is_closed': len(boundary_edges) == 0,
            'is_manifold': len(non_manifold_edges) == 0
        }

    def calculate_euler_characteristic(self, num_vertices, num_faces, num_edges):
        """計算歐拉特徵數 (V - E + F)"""
        return num_vertices - num_edges + num_faces

    def analyze_face_quality(self, vertices, faces):
        """分析面片品質"""
        areas = []
        aspect_ratios = []
        
        for face in faces:
            if len(face) == 3:  # 三角形
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                
                # 計算面積
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                areas.append(area)
                
                # 計算長寬比
                edge_lengths = [
                    np.linalg.norm(v1 - v0),
                    np.linalg.norm(v2 - v1),
                    np.linalg.norm(v0 - v2)
                ]
                max_edge = max(edge_lengths)
                min_edge = min(edge_lengths)
                aspect_ratio = max_edge / min_edge if min_edge > 0 else float('inf')
                aspect_ratios.append(aspect_ratio)
        
        return {
            'total_area': sum(areas),
            'avg_area': np.mean(areas) if areas else 0,
            'min_area': min(areas) if areas else 0,
            'max_area': max(areas) if areas else 0,
            'area_std': np.std(areas) if areas else 0,
            'avg_aspect_ratio': np.mean(aspect_ratios) if aspect_ratios else 0,
            'max_aspect_ratio': max(aspect_ratios) if aspect_ratios else 0
        }

    def find_isolated_vertices(self, vertices, faces):
        """找到孤立頂點"""
        used_vertices = set()
        for face in faces:
            used_vertices.update(face)
        
        isolated = []
        for i in range(len(vertices)):
            if i not in used_vertices:
                isolated.append(i)
        
        return isolated

    def comprehensive_analysis(self, file_path):
        """對模型進行全面分析"""
        logger.info(f"正在分析: {os.path.basename(file_path)}")
        
        # 載入模型
        vertices, faces = self.load_obj(file_path)
        
        # 基本統計
        basic_stats = {
            'vertices_count': len(vertices),
            'faces_count': len(faces),
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
        }
        
        # 連通性分析
        graph = self.build_connectivity_graph(faces)
        components = self.find_connected_components(graph, len(vertices))
        
        connectivity_stats = {
            'connected_components': len(components),
            'largest_component_size': len(max(components, key=len)) if components else 0,
            'component_size_distribution': [len(comp) for comp in components],
            'connectivity_ratio': len(max(components, key=len)) / len(vertices) if vertices.size > 0 else 0
        }
        
        # 邊流形分析
        manifold_stats = self.analyze_edge_manifold(faces)
        
        # 面片品質分析
        face_quality = self.analyze_face_quality(vertices, faces)
        
        # 孤立頂點
        isolated_vertices = self.find_isolated_vertices(vertices, faces)
        
        # 歐拉特徵數
        euler_char = self.calculate_euler_characteristic(
            len(vertices), 
            len(faces), 
            manifold_stats['total_edges']
        )
        
        # 幾何邊界框
        if len(vertices) > 0:
            bbox_min = np.min(vertices, axis=0)
            bbox_max = np.max(vertices, axis=0)
            bbox_size = bbox_max - bbox_min
            geometric_stats = {
                'bounding_box_size': bbox_size.tolist(),
                'bounding_box_volume': np.prod(bbox_size),
                'centroid': np.mean(vertices, axis=0).tolist()
            }
        else:
            geometric_stats = {
                'bounding_box_size': [0, 0, 0],
                'bounding_box_volume': 0,
                'centroid': [0, 0, 0]
            }
        
        # 整合所有分析結果
        analysis_result = {
            'file_name': os.path.basename(file_path),
            'analysis_timestamp': datetime.now().isoformat(),
            'basic_statistics': basic_stats,
            'connectivity_analysis': connectivity_stats,
            'manifold_analysis': manifold_stats,
            'face_quality_analysis': face_quality,
            'geometric_analysis': geometric_stats,
            'isolated_vertices_count': len(isolated_vertices),
            'euler_characteristic': euler_char,
            'topology_genus': (2 - euler_char) // 2 if manifold_stats['is_closed'] else None
        }
        
        return analysis_result, vertices, faces

    def filter_obj_with_analysis(self, input_path, output_path):
        """過濾 OBJ 檔案並進行對比分析"""
        # 分析原始模型
        original_analysis, vertices, faces = self.comprehensive_analysis(input_path)
        
        # 執行過濾
        graph = self.build_connectivity_graph(faces)
        components = self.find_connected_components(graph, len(vertices))
        largest_component = max(components, key=len)
        
        # 建立新的頂點和面片
        vertex_map = {}
        new_vertices = []
        new_vertex_idx = 0
        
        for i in range(len(vertices)):
            if i in largest_component:
                vertex_map[i] = new_vertex_idx
                new_vertices.append(vertices[i])
                new_vertex_idx += 1
        
        new_faces = []
        for face in faces:
            if all(v in largest_component for v in face):
                new_face = [vertex_map[v] for v in face]
                new_faces.append(new_face)
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存過濾後的模型
        with open(output_path, 'w') as f:
            for v in new_vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in new_faces:
                f.write("f " + " ".join(str(v + 1) for v in face) + "\n")
        
        # 分析處理後的模型
        processed_analysis, _, _ = self.comprehensive_analysis(output_path)
        
        # 計算處理效果
        processing_impact = self.calculate_processing_impact(original_analysis, processed_analysis)
        
        return original_analysis, processed_analysis, processing_impact

    def calculate_processing_impact(self, original, processed):
        """計算處理對模型的影響"""
        impact = {
            'vertices_reduction_rate': 1 - (processed['basic_statistics']['vertices_count'] / 
                                           original['basic_statistics']['vertices_count']),
            'faces_reduction_rate': 1 - (processed['basic_statistics']['faces_count'] / 
                                        original['basic_statistics']['faces_count']),
            'connectivity_improvement': {
                'components_reduced': original['connectivity_analysis']['connected_components'] - 
                                    processed['connectivity_analysis']['connected_components'],
                'main_component_retention': processed['connectivity_analysis']['connectivity_ratio']
            },
            'manifold_quality_change': {
                'boundary_edges_change': processed['manifold_analysis']['boundary_edges'] - 
                                       original['manifold_analysis']['boundary_edges'],
                'non_manifold_edges_change': processed['manifold_analysis']['non_manifold_edges'] - 
                                           original['manifold_analysis']['non_manifold_edges']
            },
            'topology_preservation': {
                'euler_characteristic_change': processed['euler_characteristic'] - 
                                             original['euler_characteristic'],
                'genus_change': (processed['topology_genus'] or 0) - (original['topology_genus'] or 0)
            }
        }
        
        return impact

    def generate_analysis_report(self, results, output_dir):
        """生成分析報告"""
        report_path = os.path.join(output_dir, 'geometry_analysis_report.json')
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成摘要報告
        summary_path = os.path.join(output_dir, 'geometry_analysis_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("3D模型幾何完整性分析報告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"總文件數: {len(results)}\n\n")
            
            for i, result in enumerate(results, 1):
                if 'processing_impact' in result:
                    original = result['original_analysis']
                    processed = result['processed_analysis']
                    impact = result['processing_impact']
                    
                    f.write(f"文件 {i}: {original['file_name']}\n")
                    f.write(f"原始模型統計:\n")
                    f.write(f"  頂點數: {original['basic_statistics']['vertices_count']}\n")
                    f.write(f"  面片數: {original['basic_statistics']['faces_count']}\n")
                    f.write(f"  連通分量: {original['connectivity_analysis']['connected_components']}\n")
                    f.write(f"  是否封閉: {original['manifold_analysis']['is_closed']}\n")
                    f.write(f"  是否流形: {original['manifold_analysis']['is_manifold']}\n")
                    f.write(f"  歐拉特徵數: {original['euler_characteristic']}\n")
                    
                    f.write(f"\n處理後模型統計:\n")
                    f.write(f"  頂點數: {processed['basic_statistics']['vertices_count']}\n")
                    f.write(f"  面片數: {processed['basic_statistics']['faces_count']}\n")
                    f.write(f"  連通分量: {processed['connectivity_analysis']['connected_components']}\n")
                    f.write(f"  是否封閉: {processed['manifold_analysis']['is_closed']}\n")
                    f.write(f"  是否流形: {processed['manifold_analysis']['is_manifold']}\n")
                    
                    f.write(f"\n處理效果:\n")
                    f.write(f"  頂點削減率: {impact['vertices_reduction_rate']:.2%}\n")
                    f.write(f"  面片削減率: {impact['faces_reduction_rate']:.2%}\n")
                    f.write(f"  連通分量減少: {impact['connectivity_improvement']['components_reduced']}\n")
                    f.write(f"  主體保留率: {impact['connectivity_improvement']['main_component_retention']:.2%}\n")
                    f.write("-" * 30 + "\n\n")
        
        logger.info(f"分析報告已保存: {report_path}")
        logger.info(f"摘要報告已保存: {summary_path}")

    def create_visualization(self, results, output_dir):
        """創建可視化圖表"""
        # 設置英文字體
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = True
        
        # 準備數據
        file_names = []
        original_vertices = []
        processed_vertices = []
        original_components = []
        processed_components = []
        
        # 提取有效數據
        valid_results = [r for r in results if 'processing_impact' in r]
        
        if not valid_results:
            logger.warning("No valid processing results available for plotting")
            return
        
        for i, result in enumerate(valid_results):
            file_names.append(f'Model_{i+1}')
            original_vertices.append(result['original_analysis']['basic_statistics']['vertices_count'])
            processed_vertices.append(result['processed_analysis']['basic_statistics']['vertices_count'])
            original_components.append(result['original_analysis']['connectivity_analysis']['connected_components'])
            processed_components.append(result['processed_analysis']['connectivity_analysis']['connected_components'])
        
        # 創建對比圖表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 頂點數對比
        x = range(len(file_names))
        width = 0.35
        ax1.bar([i - width/2 for i in x], original_vertices, width, label='Original', alpha=0.8)
        ax1.bar([i + width/2 for i in x], processed_vertices, width, label='Processed', alpha=0.8)
        ax1.set_title('Vertex Count Comparison', fontsize=16)
        ax1.set_xticks(x)
        ax1.set_xticklabels(file_names, rotation=45)
        ax1.legend()
        
        # 連通分量對比
        ax2.bar([i - width/2 for i in x], original_components, width, label='Original', alpha=0.8)
        ax2.bar([i + width/2 for i in x], processed_components, width, label='Processed', alpha=0.8)
        ax2.set_title('Connected Components Comparison', fontsize=16)
        ax2.set_xticks(x)
        ax2.set_xticklabels(file_names, rotation=45)
        ax2.legend()
        
        # 削減率分析
        reduction_rates = []
        for result in valid_results:
            reduction_rates.append(result['processing_impact']['vertices_reduction_rate'] * 100)
        
        ax3.bar(x, reduction_rates, alpha=0.8, color='orange')
        ax3.set_title('Vertex Reduction Rate (%)', fontsize=16)
        ax3.set_xticks(x)
        ax3.set_xticklabels(file_names, rotation=45)
        
        # 拓撲品質分析
        euler_chars = [r['original_analysis']['euler_characteristic'] for r in valid_results]
        ax4.bar(x, euler_chars, alpha=0.8, color='green')
        ax4.set_title('Euler Characteristic', fontsize=16)
        ax4.set_xticks(x)
        ax4.set_xticklabels(file_names, rotation=45)
        
        plt.tight_layout()
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        chart_path = os.path.join(output_dir, 'geometry_analysis_charts.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Analysis charts saved: {chart_path}")

    def batch_process_obj_files(self, input_dir: str, output_dir: str) -> Dict:
        """
        批量處理OBJ文件
        
        Args:
            input_dir: 輸入目錄
            output_dir: 輸出目錄
            
        Returns:
            處理結果字典
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            logger.error(f"輸入目錄不存在: {input_dir}")
            return {'success': False, 'error': f'Input directory not found: {input_dir}'}
        
        # 創建輸出目錄
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 查找OBJ文件（只在當前目錄）
        obj_files = list(input_path.glob("*.obj"))
        
        if not obj_files:
            logger.warning(f"在 {input_dir} 中未找到OBJ文件")
            return {'success': True, 'processed_files': [], 'failed_files': [], 'analysis_results': []}
        
        logger.info(f"找到 {len(obj_files)} 個OBJ文件待處理")
        
        processed_files = []
        failed_files = []
        analysis_results = []
        
        for obj_file in obj_files:
            try:
                # 創建修改後的文件名
                modified_filename = f"(modification){obj_file.name}"
                output_file = output_path / modified_filename
                
                logger.info(f"處理: {obj_file.name}")
                
                # 執行分析和過濾
                original_analysis, processed_analysis, processing_impact = self.filter_obj_with_analysis(
                    str(obj_file), str(output_file)
                )
                
                analysis_result = {
                    'original_file': str(obj_file),
                    'modified_file': str(output_file),
                    'original_analysis': original_analysis,
                    'processed_analysis': processed_analysis,
                    'processing_impact': processing_impact
                }
                
                analysis_results.append(analysis_result)
                processed_files.append({
                    'input_file': str(obj_file),
                    'output_file': str(output_file)
                })
                
                logger.info(f"✅ 處理成功: {obj_file.name}")
                
            except Exception as e:
                failed_files.append({
                    'input_file': str(obj_file),
                    'error': str(e)
                })
                logger.error(f"❌ 處理失敗: {obj_file.name} - {e}")
        
        return {
            'success': True,
            'total_files': len(obj_files),
            'processed_files': processed_files,
            'failed_files': failed_files,
            'analysis_results': analysis_results
        }

    def process_chair_models(self, base_input_dir: str, base_output_dir: str) -> Dict:
        """
        處理椅子模型目錄結構 (Chair/Chair_generation_X/)
        
        Args:
            base_input_dir: 包含Chair/Chair_generation_X的基礎輸入目錄
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
        
        # 查找Chair_generation_X目錄，過濾掉空目錄或不包含GLB/OBJ文件的目錄
        generation_dirs = []
        for d in chair_main_dir.iterdir():
            if d.is_dir() and d.name.startswith('Chair_generation_'):
                # 檢查目錄是否包含GLB文件（說明生成成功）
                glb_files = list(d.glob("*.glb"))
                obj_files = list(d.glob("*.obj"))
                
                if glb_files or obj_files:
                    generation_dirs.append(d)
                    logger.info(f"有效目錄: {d.name} (包含 {len(glb_files)} GLB, {len(obj_files)} OBJ)")
                else:
                    logger.warning(f"跳過空目錄: {d.name}")
        
        if not generation_dirs:
            logger.warning(f"在 {chair_main_dir} 中未找到有效的Chair_generation_*目錄")
            return {'success': True, 'generation_results': {}}
        
        logger.info(f"找到 {len(generation_dirs)} 個椅子生成變體目錄")
        
        generation_results = {}
        all_analysis_results = []
        
        # 創建Chair輸出目錄
        chair_output_dir = base_output_path / 'Chair'
        chair_output_dir.mkdir(parents=True, exist_ok=True)
        
        for gen_dir in sorted(generation_dirs, key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else 0):
            gen_id = gen_dir.name
            logger.info(f"處理生成變體: {gen_id}")
            
            gen_output_dir = chair_output_dir / gen_id
            
            # 處理該生成變體目錄中的所有OBJ文件
            gen_result = self.batch_process_obj_files(
                str(gen_dir), 
                str(gen_output_dir)
            )
            
            generation_results[gen_id] = gen_result
            
            # 收集分析結果
            if gen_result.get('analysis_results'):
                for analysis in gen_result['analysis_results']:
                    analysis['generation_id'] = gen_id
                    all_analysis_results.append(analysis)
            
            if gen_result['success']:
                logger.info(f"✅ {gen_id} 處理完成: 處理了 {len(gen_result['processed_files'])} 個文件")
            else:
                logger.error(f"❌ {gen_id} 處理失敗: {gen_result.get('error', 'Unknown error')}")
        
        # 生成總體分析報告
        if all_analysis_results:
            try:
                self.generate_analysis_report(all_analysis_results, str(base_output_path))
                self.create_visualization(all_analysis_results, str(base_output_path))
                logger.info("✅ 總體分析報告已生成")
            except Exception as e:
                logger.error(f"生成總體分析報告失敗: {e}")
        
        # 生成總體報告
        total_processed = sum(len(result.get('processed_files', [])) for result in generation_results.values())
        total_failed = sum(len(result.get('failed_files', [])) for result in generation_results.values())
        
        overall_report = {
            'timestamp': datetime.now().isoformat(),
            'base_input_directory': str(base_input_path),
            'base_output_directory': str(base_output_path),
            'chair_main_directory': str(chair_main_dir),
            'processed_generations': len(generation_dirs),
            'total_processed_files': total_processed,
            'total_failed_files': total_failed,
            'generation_results': generation_results
        }
        
        overall_report_file = base_output_path / 'overall_modification_report.json'
        base_output_path.mkdir(parents=True, exist_ok=True)
        with open(overall_report_file, 'w', encoding='utf-8') as f:
            json.dump(overall_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"總體處理完成: 處理了 {total_processed} 個文件，失敗 {total_failed} 個")
        logger.info(f"總體報告已保存: {overall_report_file}")
        
        return {
            'success': True,
            'processed_generations': len(generation_dirs),
            'total_processed': total_processed,
            'total_failed': total_failed,
            'generation_results': generation_results,
            'all_analysis_results': all_analysis_results,
            'report_file': str(overall_report_file)
        }

def main():
    """主函數 - 支持命令行參數"""
    parser = argparse.ArgumentParser(description='3D模型幾何分析和修改工具')
    parser.add_argument('--input_dir', type=str, required=True, help='輸入目錄路徑')
    parser.add_argument('--output_dir', type=str, required=True, help='輸出目錄路徑')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'chair'], 
                       default='chair', help='處理模式: single(單文件), batch(批量), chair(椅子結構)')
    
    args = parser.parse_args()
    
    analyzer = GeometryAnalyzer()
    
    if args.mode == 'chair':
        logger.info(f"椅子模型處理模式: {args.input_dir} -> {args.output_dir}")
        result = analyzer.process_chair_models(args.input_dir, args.output_dir)
    elif args.mode == 'batch':
        logger.info(f"批量處理模式: {args.input_dir} -> {args.output_dir}")
        result = analyzer.batch_process_obj_files(args.input_dir, args.output_dir)
        
        # 生成報告
        if result['success'] and result.get('analysis_results'):
            analyzer.generate_analysis_report(result['analysis_results'], args.output_dir)
            analyzer.create_visualization(result['analysis_results'], args.output_dir)
    else:  # single mode
        logger.info(f"單文件處理模式: {args.input_dir} -> {args.output_dir}")
        if os.path.isfile(args.input_dir) and args.input_dir.endswith('.obj'):
            try:
                # 確保輸出目錄存在
                os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
                
                original_analysis, processed_analysis, processing_impact = analyzer.filter_obj_with_analysis(
                    args.input_dir, args.output_dir
                )
                
                # 生成單文件報告
                analysis_result = {
                    'original_file': args.input_dir,
                    'modified_file': args.output_dir,
                    'original_analysis': original_analysis,
                    'processed_analysis': processed_analysis,
                    'processing_impact': processing_impact
                }
                
                report_dir = os.path.dirname(args.output_dir)
                analyzer.generate_analysis_report([analysis_result], report_dir)
                analyzer.create_visualization([analysis_result], report_dir)
                
                logger.info("✅ 單文件處理完成")
            except Exception as e:
                logger.error(f"❌ 單文件處理失敗: {e}")
        else:
            logger.error("❌ 單文件模式需要指定有效的OBJ文件路徑")
    
    if result.get('success'):
        logger.info("🎉 處理完成！")
    else:
        logger.error(f"❌ 處理失敗: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()