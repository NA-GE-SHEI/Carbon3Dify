import os
import shutil
from pathlib import Path

remove_dir_list = [
    "./image_identify",
    "./material_analysis", 
    "./3d_models",
    "./obj_models",
    "./modified_obj",
    "./geometry_analysis",
    "./bootstrap_analysis",
    "./enhanced_analysis",
    "./lca_results",
    "./workflow_reports",
    "./integrated_results",
    "./enhanced_rf_result",
    "./result"
]

remove_files_list = [
    "auto.log",
    "workflow_integration.log"
]

def cleanup_directories_and_files(remove_dirs, remove_files):
    """
    清理指定的目錄和檔案
    
    Args:
        remove_dirs: 要刪除的目錄清單
        remove_files: 要刪除的檔案清單
    """
    print("🧹 開始清理檔案和目錄...")
    print("=" * 50)
    
    # 統計變數
    deleted_dirs = 0
    deleted_files = 0
    failed_dirs = 0
    failed_files = 0
    
    # 刪除目錄
    print("📁 清理目錄:")
    for dir_path in remove_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path)
                print(f"  ✅ 已刪除目錄: {dir_path}")
                deleted_dirs += 1
            except PermissionError:
                print(f"  ❌ 權限不足，無法刪除目錄: {dir_path}")
                failed_dirs += 1
            except Exception as e:
                print(f"  ❌ 刪除目錄失敗 {dir_path}: {e}")
                failed_dirs += 1
        elif path.exists() and not path.is_dir():
            print(f"  ⚠️  路徑存在但不是目錄: {dir_path}")
        else:
            print(f"  ℹ️  目錄不存在: {dir_path}")
    
    print("\n📄 清理檔案:")
    # 刪除檔案
    for file_path in remove_files:
        path = Path(file_path)
        if path.exists() and path.is_file():
            try:
                path.unlink()
                print(f"  ✅ 已刪除檔案: {file_path}")
                deleted_files += 1
            except PermissionError:
                print(f"  ❌ 權限不足，無法刪除檔案: {file_path}")
                failed_files += 1
            except Exception as e:
                print(f"  ❌ 刪除檔案失敗 {file_path}: {e}")
                failed_files += 1
        elif path.exists() and not path.is_file():
            print(f"  ⚠️  路徑存在但不是檔案: {file_path}")
        else:
            print(f"  ℹ️  檔案不存在: {file_path}")
    
    # 顯示清理結果
    print("\n" + "=" * 50)
    print("📊 清理結果統計:")
    print(f"  成功刪除目錄: {deleted_dirs} 個")
    print(f"  成功刪除檔案: {deleted_files} 個")
    if failed_dirs > 0 or failed_files > 0:
        print(f"  刪除失敗目錄: {failed_dirs} 個")
        print(f"  刪除失敗檔案: {failed_files} 個")
    
    total_success = deleted_dirs + deleted_files
    total_failed = failed_dirs + failed_files
    
    if total_failed == 0:
        print("🎉 所有清理操作都成功完成！")
    else:
        print(f"⚠️  部分清理操作失敗，請檢查權限或檔案狀態")
    
    return {
        'deleted_dirs': deleted_dirs,
        'deleted_files': deleted_files,
        'failed_dirs': failed_dirs,
        'failed_files': failed_files
    }

def confirm_cleanup():
    """確認是否執行清理操作"""
    print("⚠️  警告：此操作將永久刪除以下目錄和檔案：")
    print("\n📁 目錄:")
    for dir_path in remove_dir_list:
        if Path(dir_path).exists():
            print(f"  - {dir_path} ✅")
        else:
            print(f"  - {dir_path} (不存在)")
    
    print("\n📄 檔案:")
    for file_path in remove_files_list:
        if Path(file_path).exists():
            print(f"  - {file_path} ✅")
        else:
            print(f"  - {file_path} (不存在)")
    
    response = input("\n是否確定要執行清理操作？(y/N): ").lower().strip()
    return response in ['y', 'yes', '是']

# 主執行區塊
if __name__ == "__main__":
    try:
        # 確認清理操作
        if confirm_cleanup():
            result = cleanup_directories_and_files(remove_dir_list, remove_files_list)
        else:
            print("🚫 清理操作已取消")
    except KeyboardInterrupt:
        print("\n🛑 用戶中斷操作")
    except Exception as e:
        print(f"❌ 執行過程中發生錯誤: {e}")
