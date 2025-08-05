#!/usr/bin/env python3
"""
Kill all server processes for the AI Healthcare HRP application.
"""

import subprocess
import sys


def kill_processes():
    """Kill all server processes."""
    print("🛑 Killing all server processes...")

    # Kill backend processes
    try:
        subprocess.run(["pkill", "-f", "uvicorn server.main_agent:app"], check=False)
        print("✅ Backend processes killed")
    except Exception as e:
        print(f"Warning: Could not kill backend processes: {e}")

    # Kill frontend processes
    try:
        subprocess.run(["pkill", "-f", "npm run dev"], check=False)
        print("✅ Frontend processes killed")
    except Exception as e:
        print(f"Warning: Could not kill frontend processes: {e}")

    # Kill any remaining node processes on our ports
    try:
        subprocess.run(["pkill", "-f", "node.*vite"], check=False)
        print("✅ Node/Vite processes killed")
    except Exception as e:
        print(f"Warning: Could not kill node processes: {e}")

    # Force kill any processes on our ports
    try:
        subprocess.run(["lsof", "-ti", ":8000"], capture_output=True, text=True)
        result = subprocess.run(
            ["lsof", "-ti", ":8000"], capture_output=True, text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid and pid.isdigit():
                    subprocess.run(["kill", "-9", pid], check=False)
                    print(f"✅ Killed process {pid} on port 8000")
    except Exception as e:
        print(f"Warning: Could not kill processes on port 8000: {e}")

    try:
        result = subprocess.run(
            ["lsof", "-ti", ":5173"], capture_output=True, text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid and pid.isdigit():
                    subprocess.run(["kill", "-9", pid], check=False)
                    print(f"✅ Killed process {pid} on port 5173")
    except Exception as e:
        print(f"Warning: Could not kill processes on port 5173: {e}")

    print("\n✅ All server processes killed!")
    print("You can now start fresh with: python3 start_clean.py")


if __name__ == "__main__":
    kill_processes()
