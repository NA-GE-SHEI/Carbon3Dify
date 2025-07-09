#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import olca_ipc as ipc
import olca_schema as o
import pandas as pd
import numpy as np
import time
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class OpenLCACalculator:
    """OpenLCA計算器類，用於與OpenLCA進行專業LCA分析 - 適配olca-ipc 2.4.0"""
    
    def __init__(self, port=8080):
        """
        初始化OpenLCA計算器
        
        Args:
            port: OpenLCA服務器端口（預設8080）
        """
        self.port = port
        self.client = None
        self.logger = logging.getLogger("OpenLCACalculator")
        
        # 預設的產品系統UUID和流映射
        self.product_system_uuid = "c22f7222-fd9c-489d-b6c5-13a9bac0c665"
        
        # 定義流的UUID和對應的CSV列名
        self.flow_mappings = {
            # 輸入流
            "d46c3fd9-95ad-4e77-ba84-a15d5fd3f5fb": "transport",     # Transport, freight
            "7409eea1-250f-4756-8c5b-2a7eb21e2a8c": "electricity",   # Electricity, low voltage
            "ae2ef150-c45a-4255-8c10-4154393c2200": "furniture_wooden", # Furniture, wooden
            # 輸出流
            "7de29c89-980d-40bd-82fd-e3670cf0be5f": "wood_product",  # Wood product
            "ea1c4d90-2c48-4539-9604-072c6b0ea06b": "waste_wood"     # Waste wood
        }
        
        # 結果保存目錄
        self.result_dir = Path("./workflow_reports/openlca_results")
        self.result_dir.mkdir(parents=True, exist_ok=True)
    
    def connect(self) -> bool:
        """
        連接到OpenLCA服務器 - 適配olca-ipc 2.4.0
        
        Returns:
            bool: 連接成功返回True，失敗返回False
        """
        try:
            # olca-ipc 2.4.0 正確的初始化方式
            self.client = ipc.Client(self.port)
            
            # 測試連接
            methods = self.client.get_descriptors(o.ImpactMethod)
            if methods:
                methods_list = list(methods)  # 轉換為列表以獲取長度
                self.logger.info(f"✅ 成功連接到OpenLCA服務器，端口: {self.port}")
                self.logger.info(f"📊 發現 {len(methods_list)} 個影響評估方法")
                return True
            else:
                self.logger.error("❌ 連接到OpenLCA但無法獲取影響評估方法")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 無法連接到OpenLCA服務器: {e}")
            # 提供詳細的錯誤診斷
            self.logger.error("請檢查：")
            self.logger.error("1. OpenLCA軟體是否正在運行")
            self.logger.error("2. IPC服務器是否已啟動 (Window → Developer tools → IPC server)")
            self.logger.error(f"3. 端口 {self.port} 是否正確且未被占用")
            return False
    
    def get_product_systems(self) -> List[Tuple[str, str]]:
        """
        獲取可用的產品系統列表
        
        Returns:
            List[Tuple[str, str]]: 產品系統ID和名稱的列表
        """
        try:
            systems = self.client.get_descriptors(o.ProductSystem)
            return [(system.id, system.name) for system in systems]
        except Exception as e:
            self.logger.error(f"❌ 獲取產品系統失敗: {e}")
            return []
    
    def get_impact_methods(self) -> List[Tuple[str, str]]:
        """
        獲取可用的影響評估方法列表
        
        Returns:
            List[Tuple[str, str]]: 影響評估方法ID和名稱的列表
        """
        try:
            methods = self.client.get_descriptors(o.ImpactMethod)
            return [(method.id, method.name) for method in methods]
        except Exception as e:
            self.logger.error(f"❌ 獲取影響評估方法失敗: {e}")
            return []
    
    def create_chair_data_csv(self, chairs_data: List[Dict]) -> str:
        """
        將椅子數據轉換為OpenLCA所需的CSV格式
        
        Args:
            chairs_data: 椅子數據列表
            
        Returns:
            str: 生成的CSV文件路徑
        """
        try:
            # 準備CSV數據
            csv_data = []
            
            for chair_data in chairs_data:
                chair_id = chair_data.get('chair_id', 'Unknown')
                
                # 從椅子數據中提取材料信息
                material_analysis = chair_data.get('material_analysis', {})
                wood_percentage = material_analysis.get('wood_percentage', 90) / 100
                estimated_weight = chair_data.get('estimated_weight', 4.0)
                
                # 計算各項數值（基於椅子重量和材料比例）
                wood_mass = estimated_weight * wood_percentage
                
                # 估算各項流的數值
                row_data = {
                    'chair_id': chair_id,
                    'wood_product': wood_mass,  # 木製品產出
                    'furniture_wooden': wood_mass * 1.1,  # 木製家具輸入（考慮加工損耗）
                    'waste_wood': wood_mass * 0.05,  # 木材廢料
                    'transport': wood_mass * 0.1,  # 運輸
                    'electricity': wood_mass * 0.5,  # 電力消耗
                    'estimated_weight': estimated_weight,
                    'wood_percentage': wood_percentage * 100
                }
                
                csv_data.append(row_data)
            
            # 創建DataFrame並保存為CSV
            df = pd.DataFrame(csv_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_file = self.result_dir / f"chair_estimation_data_{timestamp}.csv"
            
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"📄 椅子估算數據已保存為CSV: {csv_file}")
            
            return str(csv_file)
            
        except Exception as e:
            self.logger.error(f"❌ 創建椅子數據CSV失敗: {e}")
            return None
    
    def batch_calculate_chairs(self, chairs_data: List[Dict]) -> Dict:
        """
        批量計算椅子的LCA分析 - 適配olca-ipc 2.4.0
        
        Args:
            chairs_data: 椅子數據列表
            
        Returns:
            Dict: 批量計算結果
        """
        try:
            self.logger.info(f"🔄 開始批量計算 {len(chairs_data)} 個椅子的LCA分析")
            
            # 創建CSV數據文件
            csv_file = self.create_chair_data_csv(chairs_data)
            if not csv_file:
                return {'success': False, 'error': '無法創建CSV數據文件'}
            
            # 讀取CSV文件
            data = pd.read_csv(csv_file)
            self.logger.info(f"📊 成功讀取CSV文件，共 {len(data)} 行數據")
            
            # 檢查所需列
            required_columns = ["wood_product", "furniture_wooden", "waste_wood", "transport", "electricity"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.warning(f"⚠️ CSV文件缺少以下列: {', '.join(missing_columns)}")
            
            # 選擇影響評估方法
            methods = list(self.client.get_descriptors(o.ImpactMethod))
            method = None
            
            # 嘗試找到IPCC GWP方法
            for m in methods:
                if 'IPCC' in m.name or 'GWP' in m.name or 'climate' in m.name.lower():
                    method = m
                    break
            
            # 如果沒找到特定方法，使用第一個可用方法
            if not method and methods:
                method = methods[0]
            
            if not method:
                self.logger.error("❌ 沒有可用的影響評估方法")
                return {'success': False, 'error': '沒有可用的影響評估方法'}
            
            self.logger.info(f"📋 選擇的方法: {method.name}")
            
            # 開始批量處理
            all_results = pd.DataFrame()
            successful_analyses = 0
            failed_analyses = 0
            start_time = time.time()
            
            for row_index, row_data in data.iterrows():
                try:
                    chair_id = row_data.get('chair_id', f'Chair_{row_index+1}')
                    self.logger.info(f"🔄 處理椅子: {chair_id} ({row_index+1}/{len(data)})")
                    
                    # 獲取產品系統
                    product_system = self.client.get(o.ProductSystem, self.product_system_uuid)
                    
                    if not product_system:
                        self.logger.error(f"❌ 無法獲取產品系統，跳過 {chair_id}")
                        failed_analyses += 1
                        continue
                    
                    # 修改產品系統中的鏈接數值
                    modified = False
                    if hasattr(product_system, 'process_links') and product_system.process_links:
                        for link in product_system.process_links:
                            flow_id = link.flow.id if hasattr(link, 'flow') and hasattr(link.flow, 'id') else None
                            
                            if flow_id in self.flow_mappings:
                                csv_column = self.flow_mappings[flow_id]
                                if csv_column in row_data:
                                    old_value = link.amount if hasattr(link, 'amount') else "未知"
                                    link.amount = float(row_data[csv_column])
                                    modified = True
                        
                        if modified:
                            self.client.update(product_system)
                    
                    # 創建計算設置
                    setup = o.CalculationSetup()
                    setup.target = product_system
                    setup.impact_method = method
                    setup.amount = float(row_data['wood_product'])
                    
                    # 執行計算
                    result = self.client.calculate(setup)
                    result.wait_until_ready()
                    
                    # 獲取影響評估結果
                    impacts = result.get_total_impacts()
                    row_results = {}
                    
                    # 添加椅子基本信息
                    row_results['chair_id'] = chair_id
                    row_results['row_index'] = row_index + 1
                    
                    # 添加CSV數據
                    for col in required_columns:
                        if col in row_data:
                            row_results[col] = row_data[col]
                    
                    # 添加影響評估結果
                    if impacts:
                        for impact in impacts:
                            impact_name = impact.impact_category.name
                            impact_value = impact.amount
                            impact_unit = impact.impact_category.ref_unit
                            
                            row_results[f"{impact_name} ({impact_unit})"] = impact_value
                    
                    # 將結果添加到聚合DataFrame
                    all_results = pd.concat([all_results, pd.DataFrame([row_results])], ignore_index=True)
                    successful_analyses += 1
                    
                    # 釋放結果
                    result.dispose()
                    
                    # 顯示進度
                    if (row_index + 1) % 5 == 0 or row_index + 1 == len(data):
                        elapsed_time = time.time() - start_time
                        progress = (row_index + 1) / len(data) * 100
                        self.logger.info(f"📊 進度: {progress:.1f}%, 已用時間: {elapsed_time:.1f}秒")
                
                except Exception as e:
                    self.logger.error(f"❌ 處理椅子 {chair_id} 時發生錯誤: {e}")
                    failed_analyses += 1
            
            # 保存結果
            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ 批量處理完成! 成功: {successful_analyses}, 失敗: {failed_analyses}, 耗時: {elapsed_time:.1f}秒")
            
            if successful_analyses > 0:
                # 導出結果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = self.result_dir / f"openlca_batch_results_{timestamp}.csv"
                all_results.to_csv(results_file, index=False, encoding='utf-8-sig')
                
                excel_file = self.result_dir / f"openlca_batch_results_{timestamp}.xlsx"
                all_results.to_excel(excel_file, index=False)
                
                self.logger.info(f"📄 結果已保存: {results_file}")
                
                return {
                    'success': True,
                    'total_chairs': len(chairs_data),
                    'successful_analysis': successful_analyses,
                    'failed_analysis': failed_analyses,
                    'results_data': all_results,
                    'results_file': str(results_file),
                    'excel_file': str(excel_file),
                    'processing_time': elapsed_time
                }
            else:
                return {
                    'success': False,
                    'error': '所有椅子分析都失敗了',
                    'total_chairs': len(chairs_data),
                    'failed_analysis': failed_analyses
                }
                
        except Exception as e:
            self.logger.error(f"❌ 批量計算過程中發生異常: {e}")
            return {'success': False, 'error': str(e)}
    
    def close(self):
        """關閉OpenLCA連接"""
        try:
            if self.client:
                # olca-ipc 2.4.0 通常不需要顯式關閉
                self.logger.info("🔒 OpenLCA連接已關閉")
        except Exception as e:
            self.logger.warning(f"⚠️ 關閉OpenLCA連接時出錯: {e}")
