#!/usr/bin/env python3
"""
Integration Test for Single Server
==================================

This script tests that the integrated server can start and respond
to basic requests without requiring external services.
"""

import requests
import time
import subprocess
import sys
import os
from pathlib import Path

def test_server_startup():
    """Test that the server starts without critical errors."""
    print("🧪 Testing Integrated Server Startup")
    print("=" * 40)
    
    # Start server in background
    print("🚀 Starting server...")
    try:
        server_process = subprocess.Popen(
            [sys.executable, "app.py", "--port", "3001"],  # Use different port to avoid conflicts
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent
        )
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if server is responding
        print("🔍 Testing health endpoint...")
        try:
            response = requests.get("http://localhost:3001/api/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ Server started successfully!")
                print(f"   Status: {data.get('status', 'unknown')}")
                print(f"   Version: {data.get('version', 'unknown')}")
                
                components = data.get('components', {})
                for comp, status in components.items():
                    emoji = "✅" if status == "loaded" else "⚠️"
                    print(f"   {comp.title()}: {emoji} {status}")
                
                success = True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                success = False
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to server: {e}")
            success = False
        
        # Test config endpoint
        if success:
            print("\n🔧 Testing config endpoint...")
            try:
                response = requests.get("http://localhost:3001/api/config", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Config loaded: {len(data.get('personas', {}))} personas, {len(data.get('locations', []))} locations")
                else:
                    print(f"⚠️ Config endpoint failed: {response.status_code}")
            except Exception as e:
                print(f"⚠️ Config test failed: {e}")
        
        # Test basic frontend
        if success:
            print("\n🌐 Testing frontend...")
            try:
                response = requests.get("http://localhost:3001/", timeout=5)
                if response.status_code == 200:
                    print("✅ Frontend loads successfully")
                else:
                    print(f"⚠️ Frontend failed: {response.status_code}")
            except Exception as e:
                print(f"⚠️ Frontend test failed: {e}")
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        success = False
        server_process = None
    
    # Cleanup
    if server_process:
        print("\n🛑 Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("✅ Server stopped")
    
    if success:
        print("\n🎉 Integration test PASSED!")
        print("💡 The integrated server is working correctly.")
        print("🚀 You can now run: python app.py")
    else:
        print("\n❌ Integration test FAILED!")
        print("📝 Check the error messages above for troubleshooting.")
    
    return success

def test_imports():
    """Test that critical imports work."""
    print("\n🔍 Testing imports...")
    
    try:
        sys.path.append('..')
        from app import UnifiedFrontendServer, AUTOSUGGEST_AVAILABLE, SEARCH_AVAILABLE
        print("✅ Main app imports successful")
        print(f"   Autosuggest available: {'✅' if AUTOSUGGEST_AVAILABLE else '❌'}")
        print(f"   Search available: {'✅' if SEARCH_AVAILABLE else '❌'}")
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Flipkart Integrated Frontend - Integration Tests")
    print("=" * 60)
    
    # Test imports first
    imports_ok = test_imports()
    
    if imports_ok:
        # Test server startup
        server_ok = test_server_startup()
        
        if server_ok:
            print("\n🎯 ALL TESTS PASSED!")
            print("🚀 Ready to run the integrated server!")
            return True
        else:
            print("\n⚠️ Server tests failed, but imports work.")
            print("💡 You can still try running the server manually.")
            return False
    else:
        print("\n❌ Import tests failed.")
        print("📝 Install dependencies with: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)