#!/usr/bin/env python3
"""
Flipkart Frontend Startup Script
================================

This script helps start all required services in the correct order
and provides helpful status information.
"""

import subprocess
import time
import requests
import sys
import os
from pathlib import Path

class ServiceManager:
    """Manages the startup and monitoring of all required services."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.services = {
            'autosuggest': {
                'port': 5000,
                'path': self.project_root / 'autosuggest',
                'script': 'flask_server.py',
                'health_url': 'http://localhost:5000/api/health',
                'process': None
            },
            'search': {
                'port': 8000,
                'path': self.project_root / 'searchresultpage',
                'script': 'run_server.py',
                'health_url': 'http://localhost:8000/health/',
                'process': None
            },
            'frontend': {
                'port': 3000,
                'path': Path(__file__).parent,
                'script': 'app.py',
                'health_url': 'http://localhost:3000/api/health',
                'process': None
            }
        }
    
    def check_service_health(self, service_name):
        """Check if a service is healthy."""
        service = self.services[service_name]
        try:
            response = requests.get(service['health_url'], timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_service(self, service_name, timeout=30):
        """Wait for a service to become healthy."""
        print(f"⏳ Waiting for {service_name} service to start...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.check_service_health(service_name):
                print(f"✅ {service_name.title()} service is ready!")
                return True
            time.sleep(1)
        
        print(f"❌ {service_name.title()} service failed to start within {timeout} seconds")
        return False
    
    def start_service(self, service_name):
        """Start a specific service."""
        service = self.services[service_name]
        print(f"🚀 Starting {service_name} service on port {service['port']}...")
        
        # Check if already running
        if self.check_service_health(service_name):
            print(f"✅ {service_name.title()} service is already running!")
            return True
        
        # Start the service
        try:
            cwd = str(service['path'])
            cmd = [sys.executable, service['script']]
            
            service['process'] = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for service to be ready
            return self.wait_for_service(service_name)
            
        except Exception as e:
            print(f"❌ Failed to start {service_name}: {e}")
            return False
    
    def stop_service(self, service_name):
        """Stop a specific service."""
        service = self.services[service_name]
        if service['process']:
            print(f"🛑 Stopping {service_name} service...")
            service['process'].terminate()
            service['process'].wait(timeout=5)
            service['process'] = None
    
    def start_all_services(self):
        """Start all services in the correct order."""
        print("🎯 Starting Flipkart Search System")
        print("=" * 50)
        
        # Start backend services first
        success = True
        
        # Start autosuggest service
        if not self.start_service('autosuggest'):
            print("⚠️  Continuing without autosuggest service...")
            success = False
        
        # Start search service
        if not self.start_service('search'):
            print("⚠️  Continuing without search service...")
            success = False
        
        # Start frontend service
        if not self.start_service('frontend'):
            print("❌ Failed to start frontend service!")
            return False
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 All services started successfully!")
        else:
            print("⚠️  Some services failed to start, but frontend is available")
        
        print("💻 Access the application at: http://localhost:3000")
        print("🔧 Health check: http://localhost:3000/api/health")
        print("=" * 50)
        
        return True
    
    def stop_all_services(self):
        """Stop all services."""
        print("\n🛑 Stopping all services...")
        for service_name in self.services:
            self.stop_service(service_name)
        print("✅ All services stopped.")
    
    def status(self):
        """Check status of all services."""
        print("📊 Service Status")
        print("-" * 30)
        
        for service_name, service in self.services.items():
            status = "✅ Running" if self.check_service_health(service_name) else "❌ Not running"
            port = service['port']
            print(f"{service_name.title():12} (:{port}): {status}")
        
        print("-" * 30)

def main():
    """Main function with command line handling."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Flipkart Frontend Service Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py              # Start all services
  python start.py --status     # Check service status
  python start.py --stop       # Stop all services
        """
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check status of all services"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop all services"
    )
    
    args = parser.parse_args()
    
    manager = ServiceManager()
    
    try:
        if args.status:
            manager.status()
        elif args.stop:
            manager.stop_all_services()
        else:
            if manager.start_all_services():
                print("\n🔄 Services are running. Press Ctrl+C to stop all services.")
                # Keep script running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n")
                    manager.stop_all_services()
            else:
                print("❌ Failed to start services")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n")
        manager.stop_all_services()
    except Exception as e:
        print(f"❌ Error: {e}")
        manager.stop_all_services()
        sys.exit(1)

if __name__ == "__main__":
    main()