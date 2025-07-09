from openLCA import OpenLCACalculator

# 測試連接
calculator = OpenLCACalculator(port=8080)
if calculator.connect():
    print("✅ OpenLCA連接成功！")
    systems = calculator.get_product_systems()
    print(f"發現 {len(systems)} 個產品系統")
    calculator.close()
else:
    print("❌ OpenLCA連接失敗")
