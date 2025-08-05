#!/usr/bin/env python3
"""
Clean startup script for the AI Healthcare HRP application.
Handles port conflicts and process management properly.
"""

import subprocess
import time
import signal
import sys
import os
import socket
from pathlib import Path


def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def kill_process_on_port(port):
    """Kill any process using the specified port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    print(f"🛑 Killing process {pid} on port {port}")
                    subprocess.run(["kill", "-9", pid])
                    time.sleep(1)
    except Exception as e:
        print(f"Warning: Could not kill process on port {port}: {e}")


def start_backend():
    """Start the backend server."""
    print("🚀 Starting backend server...")

    # Kill any existing process on port 8000
    if is_port_in_use(8000):
        print("⚠️  Port 8000 is in use, killing existing process...")
        kill_process_on_port(8000)
        time.sleep(2)

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.main_agent:app",
            "--reload",
            "--port",
            "8000",
            "--host",
            "0.0.0.0",
        ]
    )


def start_frontend():
    """Start the frontend server."""
    print("🚀 Starting frontend server...")
    ui_dir = Path(__file__).parent / "ui"

    # Kill any existing process on port 5173
    if is_port_in_use(5173):
        print("⚠️  Port 5173 is in use, killing existing process...")
        kill_process_on_port(5173)
        time.sleep(2)

    # Check if node_modules exists
    if not (ui_dir / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=ui_dir, check=True)

    return subprocess.Popen(["npm", "run", "dev"], cwd=ui_dir)


def wait_for_server(url, max_retries=30):
    """Wait for a server to be ready."""
    import requests

    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False


def main():
    """Start both servers."""
    print("🎯 Starting AI Healthcare HRP servers...")
    print("🧹 Cleaning up any existing processes...")

    # Clean up any existing processes
    kill_process_on_port(8000)
    kill_process_on_port(5173)
    time.sleep(2)

    # Start backend
    backend_process = start_backend()
    print("⏳ Waiting for backend to start...")

    if wait_for_server("http://localhost:8000/ping"):
        print("✅ Backend server is ready!")
    else:
        print("❌ Backend server failed to start")
        backend_process.terminate()
        sys.exit(1)

    # Start frontend
    frontend_process = start_frontend()
    print("⏳ Waiting for frontend to start...")

    if wait_for_server("http://localhost:5173"):
        print("✅ Frontend server is ready!")
    else:
        print("❌ Frontend server failed to start")
        frontend_process.terminate()
        backend_process.terminate()
        sys.exit(1)

    print("\n🎉 All servers started successfully!")
    print("📱 Frontend: http://localhost:5173")
    print("🔧 Backend: http://localhost:8000")
    print("📊 API Docs: http://localhost:8000/docs")
    print("🧪 Test Page: http://localhost:5173/test.html")
    print("\nPress Ctrl+C to stop all servers")

    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)

            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend server stopped unexpectedly")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend server stopped unexpectedly")
                break

    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")

        # Stop backend
        if backend_process.poll() is None:
            backend_process.terminate()
            backend_process.wait(timeout=5)

        # Stop frontend
        if frontend_process.poll() is None:
            frontend_process.terminate()
            frontend_process.wait(timeout=5)

        print("✅ All servers stopped")


if __name__ == "__main__":
    main()
