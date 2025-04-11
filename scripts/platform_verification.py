#!/usr/bin/env python
"""
Platform Verification Logger

This script logs the verification status of the MNIST model reproduction
on different platforms and creates a report.
"""

import os
import json
import argparse
import platform
import datetime
from pathlib import Path

def get_platform_info():
    """Get detailed information about the current platform."""
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Add CUDA info if available
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except (ImportError, AttributeError):
        info["cuda_available"] = False
    
    return info

def log_verification(status="success", message="", log_dir="logs"):
    """Log the verification status with platform information."""
    platform_info = get_platform_info()
    platform_info["status"] = status
    platform_info["message"] = message
    
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a unique filename based on platform and timestamp
    system_name = platform_info["system"].lower()
    is_cuda = "cuda" if platform_info.get("cuda_available", False) else "cpu"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{system_name}_{is_cuda}_{timestamp}.json"
    
    # Write the log file
    log_path = os.path.join(log_dir, filename)
    with open(log_path, 'w') as f:
        json.dump(platform_info, f, indent=2)
    
    print(f"✅ [{system_name.upper()}] Verification {'successful' if status == 'success' else 'failed'}")
    print(f"📝 Log saved to: {log_path}")
    
    return log_path

def create_report(log_dir="logs", output_file="verification_report.md"):
    """Create a Markdown report of all verification logs."""
    log_files = list(Path(log_dir).glob("*.json"))
    
    if not log_files:
        print("⚠️ No verification logs found!")
        return
    
    logs = []
    for log_file in log_files:
        with open(log_file, 'r') as f:
            logs.append(json.load(f))
    
    # Sort logs by system and timestamp
    logs.sort(key=lambda x: (x["system"], x["timestamp"]))
    
    # Create report
    with open(output_file, 'w') as f:
        f.write("# MNIST Model Reproduction Verification Report\n\n")
        f.write(f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Platform | Status | Timestamp | CUDA | Python Version |\n")
        f.write("|----------|--------|-----------|------|---------------|\n")
        
        for log in logs:
            cuda_info = f"{log.get('cuda_version', 'N/A')}" if log.get("cuda_available", False) else "N/A"
            timestamp = datetime.datetime.fromisoformat(log["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
            status_emoji = "✅" if log["status"] == "success" else "❌"
            
            f.write(f"| {log['system']} {log['machine']} | {status_emoji} | {timestamp} | {cuda_info} | {log['python_version']} |\n")
        
        f.write("\n## Detailed Information\n\n")
        
        for i, log in enumerate(logs, 1):
            f.write(f"### {i}. {log['system']} ({log['machine']})\n\n")
            f.write(f"- **Status**: {'Success' if log['status'] == 'success' else 'Failed'}\n")
            f.write(f"- **Timestamp**: {datetime.datetime.fromisoformat(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **OS Version**: {log['release']} ({log['version']})\n")
            f.write(f"- **Python Version**: {log['python_version']}\n")
            
            if log.get("cuda_available", False):
                f.write(f"- **CUDA Available**: Yes\n")
                f.write(f"- **CUDA Version**: {log.get('cuda_version', 'N/A')}\n")
                f.write(f"- **cuDNN Version**: {log.get('cudnn_version', 'N/A')}\n")
                f.write(f"- **GPU**: {log.get('gpu_name', 'N/A')}\n")
            else:
                f.write(f"- **CUDA Available**: No\n")
            
            if log["message"]:
                f.write(f"- **Message**: {log['message']}\n")
            
            f.write("\n")
    
    print(f"📊 Verification report created: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Log platform verification status")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Log command
    log_parser = subparsers.add_parser("log", help="Log verification status")
    log_parser.add_argument("--status", choices=["success", "failure"], default="success",
                           help="Verification status (success or failure)")
    log_parser.add_argument("--message", default="", help="Additional message")
    log_parser.add_argument("--log-dir", default="logs", help="Directory to store logs")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Create verification report")
    report_parser.add_argument("--log-dir", default="logs", help="Directory with log files")
    report_parser.add_argument("--output", default="verification_report.md", 
                              help="Output report file (Markdown)")
    
    args = parser.parse_args()
    
    if args.command == "log":
        log_verification(args.status, args.message, args.log_dir)
    elif args.command == "report":
        create_report(args.log_dir, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 