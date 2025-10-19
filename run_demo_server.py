#!/usr/bin/env python3
"""
Demo Server Script for Facial Emotion Recognition
This script helps you run the app for external demonstrations
"""

import socket
import subprocess
import sys
from pathlib import Path


def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    print("🚀 Facial Emotion Recognition - Demo Server")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ Error: streamlit_app.py not found!")
        print("Please run this script from the project root directory.")
        return

    # Get network info
    local_ip = get_local_ip()
    port = 8501

    print(f"📡 Local IP: {local_ip}")
    print(f"🌐 Port: {port}")
    print(f"🔗 Local URL: http://localhost:{port}")
    print(f"🌍 Network URL: http://{local_ip}:{port}")
    print()

    # Ask user what they want to do
    print("Choose an option:")
    print("1. Run locally only (localhost)")
    print("2. Run for network access (external users)")
    print("3. Just show network info")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        print("🏠 Starting local server...")
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.port",
            str(port),
        ]

    elif choice == "2":
        print("🌐 Starting network server...")
        print("⚠️  WARNING: This makes your app accessible to anyone on your network!")
        print("   Make sure your firewall is configured properly.")
        print()
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            str(port),
        ]

    elif choice == "3":
        print("📋 Network Information:")
        print(f"   Local URL: http://localhost:{port}")
        print(f"   Network URL: http://{local_ip}:{port}")
        print()
        print("To share with others:")
        print(
            f"   1. Run: streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port {port}"
        )
        print(f"   2. Give them this URL: http://{local_ip}:{port}")
        return

    else:
        print("❌ Invalid choice!")
        return

    print()
    print("🚀 Starting Streamlit server...")
    print("Press Ctrl+C to stop the server")
    print()

    try:
        # Start the server
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
