import os
import pandas as pd
import olca
import json
import logging
import requests
from typing import Dict, List, Optional, Any, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OpenLCA-Calc")

class OpenLCACalculator:
    """OpenLCA計算核心功能模組"""
    
    def __init__(self, port: int = 8080):
        """初始化OpenLCA連接"""
        self.client = None
        self.port = port
        self.url = f'http://localhost:{port}'
        self.next_id = 1
    
    def connect(self) -> bool:
        """連接到OpenLCA IPC服務器"""
        try:
            logger.info(f"連接到OpenLCA IPC服務器 (端口:{self.port})...")
            self.client = olca.Client(self.port)
            logger.info("連接成功")
            return True
        except Exception as e:
            logger.error(f"連接失敗: {str(e)}")
            return False
    
    def _json_rpc_call(self, method: str, params=None) -> Tuple[Any, Optional[str]]:
        """發送JSON-RPC請求
        
        Args:
            method: 方法名
            params: 參數對象
            
        Returns:
            (結果, 錯誤訊息) Tuple
        """
        req = {
            'jsonrpc': '2.0',
            'id': self.next_id,
            'method': method
        }
        if params is not None:
            req['params'] = params
        
        self.next_id += 1
        
        try:
            resp = requests.post(self.url, json=req).json()
            
            err = resp.get('error')
            if err is not None:
                err_msg = f"{err.get('code')}: {err.get('message')}"
                return None, err_msg
            
            result = resp.get('result')
            if result is None:
                return None, "No error and no result: invalid JSON-RPC response"
            
            return result, None
        except Exception as e:
            return None, str(e)
    
    def get_product_systems(self) -> List[Tuple[str, str]]:
        """獲取所有可用的產品系統"""
        if not self.client:
            logger.error("未連接到OpenLCA")
            return []
        
        try:
            systems = list(self.client.get_descriptors(olca.ProductSystem))
            return [(s.id, s.name) for s in systems]
        except Exception as e:
            logger.error(f"獲取產品系統失敗: {str(e)}")
            return []
    
    def get_product_system_details(self, product_system_id) -> Dict:
        """獲取產品系統的詳細信息
        
        Args:
            product_system_id: 產品系統ID
            
        Returns:
            產品系統詳細信息的字典
        """
        if not self.client:
            logger.error("未連接到OpenLCA")
            return {}
        
        try:
            # 直接獲取完整的產品系統對象
            product_system = self.client.get(olca.ProductSystem, product_system_id)
            
            # 如果使用直接API失敗，嘗試使用JSON-RPC
            if not product_system:
                logger.info("嘗試使用JSON-RPC獲取產品系統")
                product_system, err = self._json_rpc_call('get/model', {
                    '@model': 'ProductSystem',
                    '@id': product_system_id
                })
                
                if err:
                    logger.error(f"獲取產品系統失敗: {err}")
                    return {}
            
            if product_system:
                logger.info(f"成功獲取產品系統: {product_system.name if hasattr(product_system, 'name') else '未知名稱'}")
                
                # 將產品系統信息轉換為標準格式的字典
                system_info = self._convert_product_system_to_dict(product_system)
                return system_info
            else:
                logger.error("產品系統返回為空")
                return {}
            
        except Exception as e:
            logger.error(f"獲取產品系統時出錯: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _convert_product_system_to_dict(self, product_system) -> Dict:
        """將產品系統對象轉換為字典格式
        
        處理對象型和字典型的產品系統，返回標準格式的字典
        """
        result = {}
        
        # 檢查是否是字典型
        if isinstance(product_system, dict):
            result = {
                'id': product_system.get('@id', ''),
                'name': product_system.get('name', ''),
                'description': product_system.get('description', ''),
                'processes_count': len(product_system.get('processes', [])),
                'links_count': len(product_system.get('processLinks', [])),
                'processes': []
            }
            
            # 獲取參考流信息
            ref_exchange = product_system.get('referenceExchange', {})
            ref_flow = ref_exchange.get('flow', {})
            result['reference_flow'] = {
                'id': ref_flow.get('@id', ''),
                'name': ref_flow.get('name', ''),
                'amount': ref_exchange.get('amount', 0),
                'unit': ref_exchange.get('unit', '')
            }
            
            # 獲取進程列表
            for proc in product_system.get('processes', [])[:10]:  # 只取前10個
                process_info = {
                    'id': proc.get('@id', ''),
                    'name': proc.get('name', '')
                }
                result['processes'].append(process_info)
        else:
            # 對象型處理
            result = {
                'id': getattr(product_system, 'id', ''),
                'name': getattr(product_system, 'name', ''),
                'description': getattr(product_system, 'description', ''),
                'processes_count': len(getattr(product_system, 'processes', [])),
                'links_count': len(getattr(product_system, 'process_links', [])),
                'processes': []
            }
            
            # 獲取參考流信息
            ref_exchange = getattr(product_system, 'reference_exchange', None)
            if ref_exchange:
                ref_flow = getattr(ref_exchange, 'flow', None)
                result['reference_flow'] = {
                    'id': getattr(ref_flow, 'id', '') if ref_flow else '',
                    'name': getattr(ref_flow, 'name', '') if ref_flow else '',
                    'amount': getattr(ref_exchange, 'amount', 0),
                    'unit': getattr(ref_exchange, 'unit', '')
                }
            else:
                result['reference_flow'] = {
                    'id': '',
                    'name': '',
                    'amount': 0,
                    'unit': ''
                }
            
            # 獲取進程列表
            processes = getattr(product_system, 'processes', [])
            for proc in processes[:10]:  # 只取前10個
                process_info = {
                    'id': getattr(proc, 'id', ''),
                    'name': getattr(proc, 'name', '')
                }
                result['processes'].append(process_info)
        
        return result
    
    def get_process_details(self, process_id) -> Dict:
        """獲取進程的詳細信息
        
        Args:
            process_id: 進程ID
            
        Returns:
            進程詳細信息的字典
        """
        if not self.client:
            logger.error("未連接到OpenLCA")
            return {}
        
        try:
            # 直接獲取完整的進程對象
            process = self.client.get(olca.Process, process_id)
            
            # 如果使用直接API失敗，嘗試使用JSON-RPC
            if not process:
                logger.info("嘗試使用JSON-RPC獲取進程")
                process, err = self._json_rpc_call('get/model', {
                    '@model': 'Process',
                    '@id': process_id
                })
                
                if err:
                    logger.error(f"獲取進程失敗: {err}")
                    return {}
            
            if process:
                logger.info(f"成功獲取進程: {process.name if hasattr(process, 'name') else '未知名稱'}")
                
                # 將進程信息轉換為標準格式的字典
                return self._convert_process_to_dict(process)
            else:
                logger.error("進程返回為空")
                return {}
            
        except Exception as e:
            logger.error(f"獲取進程時出錯: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _convert_process_to_dict(self, process) -> Dict:
        """將進程對象轉換為字典格式
        
        處理對象型和字典型的進程，返回標準格式的字典
        """
        result = {}
        
        # 檢查是否是字典型
        if isinstance(process, dict):
            result = {
                'id': process.get('@id', ''),
                'name': process.get('name', ''),
                'description': process.get('description', ''),
                'category': process.get('category', {}).get('name', ''),
                'inputs': [],
                'outputs': []
            }
            
            # 獲取輸入和輸出
            for exchange in process.get('exchanges', []):
                is_input = exchange.get('input', False)
                flow = exchange.get('flow', {})
                
                exchange_info = {
                    'id': flow.get('@id', ''),
                    'name': flow.get('name', ''),
                    'amount': exchange.get('amount', 0),
                    'unit': exchange.get('unit', ''),
                    'flow_type': flow.get('flowType', '')
                }
                
                if is_input:
                    result['inputs'].append(exchange_info)
                else:
                    result['outputs'].append(exchange_info)
                
        else:
            # 對象型處理
            result = {
                'id': getattr(process, 'id', ''),
                'name': getattr(process, 'name', ''),
                'description': getattr(process, 'description', ''),
                'category': getattr(getattr(process, 'category', None), 'name', '') if getattr(process, 'category', None) else '',
                'inputs': [],
                'outputs': []
            }
            
            # 獲取輸入和輸出
            exchanges = getattr(process, 'exchanges', [])
            for exchange in exchanges:
                is_input = getattr(exchange, 'input', False)
                flow = getattr(exchange, 'flow', None)
                
                exchange_info = {
                    'id': getattr(flow, 'id', '') if flow else '',
                    'name': getattr(flow, 'name', '') if flow else '',
                    'amount': getattr(exchange, 'amount', 0),
                    'unit': getattr(exchange, 'unit', ''),
                    'flow_type': getattr(flow, 'flow_type', '') if flow else ''
                }
                
                if is_input:
                    result['inputs'].append(exchange_info)
                else:
                    result['outputs'].append(exchange_info)
        
        return result
    
    def get_impact_methods(self) -> List[Tuple[str, str]]:
        """獲取所有可用的影響評估方法"""
        if not self.client:
            logger.error("未連接到OpenLCA")
            return []
        
        try:
            methods = list(self.client.get_descriptors(olca.ImpactMethod))
            return [(m.id, m.name) for m in methods]
        except Exception as e:
            logger.error(f"獲取影響評估方法失敗: {str(e)}")
            return []
    
    def run_calculation(self, product_system_id: str, amount: float = 1.0, 
                        impact_method_id: str = None) -> Optional[Any]:
        """執行產品系統計算"""
        if not self.client:
            logger.error("未連接到OpenLCA")
            return None
        
        try:
            # 設置計算
            setup = olca.CalculationSetup()
            # 修正：使用正確的計算類型枚舉
            setup.calculation_type = olca.CalculationType.CONTRIBUTION_ANALYSIS
            setup.product_system = olca.ref(olca.ProductSystem, product_system_id)
            setup.amount = amount
            
            # 如果提供了影響評估方法ID，則使用它
            if impact_method_id:
                setup.impact_method = olca.ref(olca.ImpactMethod, impact_method_id)
            
            logger.info(f"執行產品系統計算 (系統ID: {product_system_id}, 數量: {amount})")
            result = self.client.calculate(setup)
            
            # 等待計算完成
            if hasattr(result, 'wait_until_ready'):
                result.wait_until_ready()
            
            logger.info("計算完成")
            
            # 輸出結果ID
            if hasattr(result, 'uid'):
                logger.info(f"結果ID: {result.uid}")
            
            return result
        except Exception as e:
            logger.error(f"計算失敗: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_inventory_results(self, result) -> Dict[str, pd.DataFrame]:
        """獲取庫存分析(LCI)結果"""
        if not result:
            logger.error("結果為空")
            return {'inputs': pd.DataFrame(), 'outputs': pd.DataFrame()}
        
        try:
            # 打印結果UID信息
            logger.info(f"結果UID: {result.uid if hasattr(result, 'uid') else '無UID'}")
            
            # 使用直接方法獲取總流量
            try:
                total_flows = result.get_total_flows()
                logger.info(f"使用get_total_flows方法獲取到 {len(total_flows) if total_flows else 0} 個流")
            except Exception as e:
                logger.error(f"直接調用get_total_flows失敗: {str(e)}")
                # 使用JSON-RPC
                params = {'@id': str(result.uid)} if hasattr(result, 'uid') else {}
                total_flows, err = self._json_rpc_call('get/total-flows', params)
                if err:
                    logger.error(f"JSON-RPC獲取總流量失敗: {err}")
                    total_flows = []
                else:
                    logger.info(f"使用JSON-RPC獲取到 {len(total_flows) if total_flows else 0} 個流")
            
            if not total_flows:
                # 嘗試使用get_envi_flows
                try:
                    envi_flows = result.get_envi_flows()
                    if envi_flows:
                        logger.info(f"使用get_envi_flows獲取到 {len(envi_flows)} 個環境流")
                        # 將環境流轉換為總流量格式
                        total_flows = []
                        for flow in envi_flows:
                            total_flow = {
                                'amount': getattr(flow, 'amount', 0),
                                'envi_flow': {
                                    'flow': {
                                        '@id': getattr(getattr(flow, 'flow', None), 'id', ''),
                                        'name': getattr(getattr(flow, 'flow', None), 'name', '')
                                    },
                                    'is_input': getattr(flow, 'is_input', False),
                                    'unit': getattr(flow, 'unit', '')
                                }
                            }
                            total_flows.append(total_flow)
                except Exception as e:
                    logger.error(f"使用get_envi_flows失敗: {str(e)}")
                    # 最後嘗試使用JSON-RPC
                    params = {'@id': str(result.uid)} if hasattr(result, 'uid') else {}
                    envi_flows, err = self._json_rpc_call('get/envi-flows', params)
                    if err:
                        logger.error(f"JSON-RPC獲取環境流失敗: {err}")
                    elif envi_flows:
                        logger.info(f"使用JSON-RPC獲取到 {len(envi_flows)} 個環境流")
                        # 將環境流轉換為總流量格式
                        total_flows = []
                        for flow in envi_flows:
                            total_flow = {
                                'amount': flow.get('amount', 0),
                                'envi_flow': {
                                    'flow': {
                                        '@id': flow.get('flow', {}).get('@id', ''),
                                        'name': flow.get('flow', {}).get('name', '')
                                    },
                                    'is_input': flow.get('is_input', False),
                                    'unit': flow.get('unit', '')
                                }
                            }
                            total_flows.append(total_flow)
            
            if not total_flows:
                logger.warning("沒有找到任何流數據")
                return {'inputs': pd.DataFrame(), 'outputs': pd.DataFrame()}
            
            # 將流分為輸入和輸出
            inputs = []
            outputs = []
            
            for flow in total_flows:
                # 檢查是否是對象或字典
                if isinstance(flow, dict):
                    # 字典類型處理
                    is_input = flow.get('envi_flow', {}).get('is_input', False)
                    if is_input:
                        inputs.append(flow)
                    else:
                        outputs.append(flow)
                else:
                    # 對象類型處理
                    envi_flow = getattr(flow, 'envi_flow', None)
                    is_input = getattr(envi_flow, 'is_input', False) if envi_flow else False
                    
                    if is_input:
                        inputs.append(flow)
                    else:
                        outputs.append(flow)
            
            logger.info(f"分離出 {len(inputs)} 個輸入流和 {len(outputs)} 個輸出流")
            
            # 轉換為DataFrame - 對象型和字典型分別處理
            inputs_df = self._convert_flows_to_df(inputs, 'Input')
            outputs_df = self._convert_flows_to_df(outputs, 'Output')
            
            return {
                'inputs': inputs_df,
                'outputs': outputs_df
            }
        except Exception as e:
            logger.error(f"獲取庫存結果失敗: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'inputs': pd.DataFrame(), 'outputs': pd.DataFrame()}
    
    def _convert_flows_to_df(self, flows, direction):
        """將流轉換為DataFrame"""
        if not flows:
            return pd.DataFrame()
        
        rows = []
        for flow in flows:
            row = {'direction': direction}
            
            # 檢查是否是對象或字典
            if isinstance(flow, dict):
                # 字典類型處理
                envi_flow = flow.get('envi_flow', {})
                flow_obj = envi_flow.get('flow', {})
                
                row['flow_id'] = flow_obj.get('@id')
                row['flow_name'] = flow_obj.get('name')
                row['amount'] = flow.get('amount')
                row['unit'] = envi_flow.get('unit')
            else:
                # 對象類型處理
                envi_flow = getattr(flow, 'envi_flow', None)
                flow_obj = getattr(envi_flow, 'flow', None) if envi_flow else None
                
                row['flow_id'] = getattr(flow_obj, 'id', None) if flow_obj else None
                row['flow_name'] = getattr(flow_obj, 'name', None) if flow_obj else None
                row['amount'] = getattr(flow, 'amount', None)
                row['unit'] = getattr(envi_flow, 'unit', None) if envi_flow else None
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_impact_results(self, result) -> pd.DataFrame:
        """獲取影響評估(LCIA)結果"""
        if not result:
            logger.error("結果為空")
            return pd.DataFrame()
        
        try:
            # 打印結果UID信息
            logger.info(f"結果UID: {result.uid if hasattr(result, 'uid') else '無UID'}")
            
            # 直接使用result對象的方法
            try:
                total_impacts = result.get_total_impacts()
                logger.info(f"使用get_total_impacts方法獲取到 {len(total_impacts) if total_impacts else 0} 個影響結果")
            except Exception as e:
                logger.error(f"直接調用get_total_impacts失敗: {str(e)}")
                # 嘗試使用JSON-RPC
                params = {'@id': str(result.uid)} if hasattr(result, 'uid') else {}
                total_impacts, err = self._json_rpc_call('get/total-impacts', params)
                if err:
                    logger.error(f"JSON-RPC獲取影響評估結果失敗: {err}")
                    return pd.DataFrame()
                else:
                    logger.info(f"使用JSON-RPC獲取到 {len(total_impacts) if total_impacts else 0} 個影響結果")
            
            if not total_impacts:
                logger.warning("沒有找到影響評估結果")
                return pd.DataFrame()
            
            # 轉換為DataFrame - 對象型和字典型分別處理
            rows = []
            for impact in total_impacts:
                row = {}
                
                # 檢查是否是對象或字典
                if isinstance(impact, dict):
                    # 字典類型處理
                    category = impact.get('impact_category', {})
                    
                    row['category_id'] = category.get('@id')
                    row['category_name'] = category.get('name')
                    row['amount'] = impact.get('amount')
                    row['unit'] = impact.get('unit')
                else:
                    # 對象類型處理
                    category = getattr(impact, 'impact_category', None)
                    
                    row['category_id'] = getattr(category, 'id', None) if category else None
                    row['category_name'] = getattr(category, 'name', None) if category else None
                    row['amount'] = getattr(impact, 'amount', None)
                    row['unit'] = getattr(impact, 'unit', None)
                
                rows.append(row)
            
            impact_df = pd.DataFrame(rows)
            return impact_df
        except Exception as e:
            logger.error(f"獲取影響評估結果失敗: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def close_result(self, result) -> bool:
        """釋放計算結果資源"""
        if not result:
            return False
        
        try:
            # 使用dispose方法釋放資源
            result.dispose()
            logger.info("已釋放結果資源")
            return True
        except Exception as e:
            logger.error(f"釋放結果資源失敗: {str(e)}")
            return False
    
    def close(self):
        """關閉連接"""
        if self.client:
            self.client.close()
            logger.info("關閉OpenLCA連接")


# 使用示例
def example_usage():
    """示範如何使用OpenLCACalculator類"""
    # 初始化計算器
    calc = OpenLCACalculator()
    
    # 連接到OpenLCA
    if not calc.connect():
        return
    
    try:
        # 獲取所有產品系統並顯示
        systems = calc.get_product_systems()
        print("\n可用的產品系統:")
        for i, (sys_id, sys_name) in enumerate(systems):
            print(f"{i+1}. {sys_name} (ID: {sys_id})")
        
        # 選擇產品系統
        system_index = int(input("選擇產品系統編號: ")) - 1
        if system_index < 0 or system_index >= len(systems):
            print("無效的選擇")
            return
        
        system_id, system_name = systems[system_index]
        print(f"已選擇: {system_name}")
        
        # 選擇操作模式
        print("\n選擇操作模式:")
        print("1. 獲取產品系統詳細信息")
        print("2. 執行計算")
        mode = int(input("請選擇操作模式 (1 或 2): "))
        
        if mode == 1:
            # 獲取產品系統詳細信息
            print("\n正在獲取產品系統詳細信息...")
            system_details = calc.get_product_system_details(system_id)
            
            if system_details:
                print("\n產品系統基本信息:")
                print(f"ID: {system_details.get('id', '')}")
                print(f"名稱: {system_details.get('name', '')}")
                print(f"描述: {system_details.get('description', '')}")
                
                ref_flow = system_details.get('reference_flow', {})
                print(f"\n參考流:")
                print(f"ID: {ref_flow.get('id', '')}")
                print(f"名稱: {ref_flow.get('name', '')}")
                print(f"數量: {ref_flow.get('amount', '')}")
                print(f"單位: {ref_flow.get('unit', '')}")
                
                print(f"\n進程數量: {system_details.get('processes_count', 0)}")
                print(f"連接數量: {system_details.get('links_count', 0)}")
                
                processes = system_details.get('processes', [])
                if processes:
                    print("\n產品系統中的進程 (前10個):")
                    for i, proc in enumerate(processes):
                        print(f"{i+1}. {proc.get('name', '未知進程名')} (ID: {proc.get('id', '未知ID')})")
                    
                    # 可選: 獲取特定進程的詳細信息
                    proc_index = int(input("\n選擇進程編號查看詳細信息 (0 表示不查看): ")) - 1
                    if proc_index >= 0 and proc_index < len(processes):
                        proc_id = processes[proc_index]['id']
                        proc_name = processes[proc_index]['name']
                        print(f"\n正在獲取進程 '{proc_name}' 的詳細信息...")
                        
                        proc_details = calc.get_process_details(proc_id)
                        if proc_details:
                            print(f"\n進程 '{proc_name}' 詳細信息:")
                            print(f"ID: {proc_details.get('id', '')}")
                            print(f"名稱: {proc_details.get('name', '')}")
                            print(f"描述: {proc_details.get('description', '')}")
                            print(f"類別: {proc_details.get('category', '')}")
                            
                            inputs = proc_details.get('inputs', [])
                            print(f"\n輸入流 ({len(inputs)}):")
                            for i, input_flow in enumerate(inputs[:5]):  # 只顯示前5個
                                print(f"{i+1}. {input_flow.get('name', '')} - {input_flow.get('amount', 0)} {input_flow.get('unit', '')}")
                            if len(inputs) > 5:
                                print(f"... 還有 {len(inputs) - 5} 個輸入流")
                            
                            outputs = proc_details.get('outputs', [])
                            print(f"\n輸出流 ({len(outputs)}):")
                            for i, output_flow in enumerate(outputs[:5]):  # 只顯示前5個
                                print(f"{i+1}. {output_flow.get('name', '')} - {output_flow.get('amount', 0)} {output_flow.get('unit', '')}")
                            if len(outputs) > 5:
                                print(f"... 還有 {len(outputs) - 5} 個輸出流")
        
        elif mode == 2:
            # 獲取影響評估方法
            methods = calc.get_impact_methods()
            
            # 選擇影響評估方法
            method_id = None
            if methods:
                print("\n可用的影響評估方法:")
                print("0. 不使用影響評估方法")
                for i, (m_id, m_name) in enumerate(methods):
                    print(f"{i+1}. {m_name}")
                
                method_index = int(input("選擇影響評估方法編號: ")) - 1
                if method_index >= 0 and method_index < len(methods):
                    method_id, method_name = methods[method_index]
                    print(f"已選擇: {method_name}")
            
            # 設置計算量
            amount = float(input("輸入計算量 (默認為1.0): ") or "1.0")
            
            # 執行計算
            result = calc.run_calculation(system_id, amount, method_id)
            if not result:
                return
            
            # 打印結果對象信息
            print("\n結果對象信息:")
            print(f"類型: {type(result)}")
            print(f"可用方法: {[m for m in dir(result) if not m.startswith('_')]}")
            print(f"結果ID: {result.uid if hasattr(result, 'uid') else 'Unknown'}")
            
            # 先打印調試信息
            print("\n調試結果對象:")
            print(f"結果類: {result.__class__.__name__}")
            print(f"UID存在: {hasattr(result, 'uid')}")
            if hasattr(result, 'uid'):
                print(f"UID值: '{result.uid}'")
                print(f"UID類型: {type(result.uid)}")
            
            # 嘗試直接調用方法
            try:
                flows = result.get_total_flows()
                print(f"直接調用get_total_flows: {len(flows) if flows else 0} 個流")
            except Exception as e:
                print(f"直接調用get_total_flows錯誤: {str(e)}")
            
            # 獲取庫存結果
            inventory = calc.get_inventory_results(result)
            
            # 顯示輸入與輸出流
            print("\n===== 輸入流 (前5項) =====")
            print(inventory['inputs'].head() if not inventory['inputs'].empty else "無輸入流數據")
            
            print("\n===== 輸出流 (前5項) =====")
            print(inventory['outputs'].head() if not inventory['outputs'].empty else "無輸出流數據")
            
            # 獲取影響評估結果
            if method_id:
                lcia_results = calc.get_impact_results(result)
                
                if not lcia_results.empty:
                    print("\n===== 影響評估結果 =====")
                    print(lcia_results)
            
            # 保存結果
            output_dir = f"./results_{system_name.replace(' ', '_')}"
            os.makedirs(output_dir, exist_ok=True)
            
            if not inventory['inputs'].empty:
                inventory['inputs'].to_excel(f"{output_dir}/inputs.xlsx", index=False)
                print(f"輸入流已保存到: {output_dir}/inputs.xlsx")
            
            if not inventory['outputs'].empty:
                inventory['outputs'].to_excel(f"{output_dir}/outputs.xlsx", index=False)
                print(f"輸出流已保存到: {output_dir}/outputs.xlsx")
            
            if method_id and not lcia_results.empty:
                lcia_results.to_excel(f"{output_dir}/impact_results.xlsx", index=False)
                print(f"影響評估結果已保存到: {output_dir}/impact_results.xlsx")
            
            print(f"\n結果已保存到: {output_dir}")
            
            # 釋放結果資源
            calc.close_result(result)
        
        else:
            print("無效的選擇")
    
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 關閉連接
        calc.close()


if __name__ == "__main__":
    example_usage()
