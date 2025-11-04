"""
测试脚本 - 验证Web应用的基本功能
"""

import requests
import time
import sys
from pathlib import Path


def test_health_check(base_url="http://localhost:8000"):
    """测试健康检查端点"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   - Status: {data['status']}")
            print(f"   - GPU available: {data['gpu_available']}")
            print(f"   - Active tasks: {data['active_tasks']}")
            print(f"   - Total tasks: {data['total_tasks']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False


def test_main_page(base_url="http://localhost:8000"):
    """测试主页"""
    print("\n🔍 Testing main page...")
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ Main page accessible")
            return True
        else:
            print(f"❌ Main page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Main page error: {str(e)}")
        return False


def test_api_docs(base_url="http://localhost:8000"):
    """测试API文档"""
    print("\n🔍 Testing API documentation...")
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✅ API documentation accessible")
            return True
        else:
            print(f"❌ API docs failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API docs error: {str(e)}")
        return False


def test_file_upload(base_url="http://localhost:8000", test_file=None):
    """测试文件上传（如果提供了测试文件）"""
    if test_file is None:
        print("\n⏭️  Skipping file upload test (no test file provided)")
        return True
    
    print(f"\n🔍 Testing file upload with: {test_file}")
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (Path(test_file).name, f)}
            response = requests.post(f"{base_url}/api/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                task_id = data.get('task_id')
                print(f"✅ File uploaded successfully")
                print(f"   - Task ID: {task_id}")
                
                # 监控任务状态
                print("\n📊 Monitoring task progress...")
                return monitor_task(base_url, task_id)
            else:
                print(f"❌ Upload failed: {data}")
                return False
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return False


def monitor_task(base_url, task_id, max_wait=1800):
    """监控任务进度"""
    start_time = time.time()
    last_progress = -1
    
    while True:
        try:
            response = requests.get(f"{base_url}/api/task/{task_id}")
            if response.status_code == 200:
                task = response.json()
                status = task.get('status')
                progress = task.get('progress', 0)
                message = task.get('message', '')
                
                # 只在进度变化时打印
                if progress != last_progress:
                    print(f"   [{progress}%] {message}")
                    last_progress = progress
                
                if status == 'completed':
                    print(f"\n✅ Task completed successfully!")
                    print(f"   - Report URL: {task.get('report_url')}")
                    return True
                elif status == 'failed':
                    print(f"\n❌ Task failed: {task.get('error')}")
                    return False
                
                # 检查超时
                if time.time() - start_time > max_wait:
                    print(f"\n⏱️  Task timeout (>{max_wait}s)")
                    return False
                
                # 等待2秒后再次检查
                time.sleep(2)
            else:
                print(f"❌ Failed to get task status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error monitoring task: {str(e)}")
            return False


def main():
    """主测试函数"""
    print("=" * 80)
    print("CpGPT Web Application Test Suite")
    print("=" * 80)
    
    base_url = "http://localhost:8000"
    
    # 检查服务器是否运行
    print("\n🔍 Checking if server is running...")
    try:
        requests.get(base_url, timeout=2)
        print("✅ Server is running")
    except:
        print("❌ Server is not running!")
        print("   Please start the server first:")
        print("   bash webapp/start_server.sh")
        sys.exit(1)
    
    # 运行测试
    results = []
    
    results.append(("Health Check", test_health_check(base_url)))
    results.append(("Main Page", test_main_page(base_url)))
    results.append(("API Docs", test_api_docs(base_url)))
    
    # 如果提供了测试文件，运行上传测试
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        results.append(("File Upload", test_file_upload(base_url, test_file)))
    
    # 打印总结
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

