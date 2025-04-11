import os
import hashlib
import argparse
import json
import sys

def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def update_hash_json(target_files):
    """Update hash.json with the hashes of target files"""
    hash_file_path = "hash.json"
    hash_data = {}
    
    # Try to load existing hash data
    if os.path.exists(hash_file_path) and os.path.getsize(hash_file_path) > 0:
        try:
            with open(hash_file_path, 'r') as f:
                hash_data = json.load(f)
                print(f"📖 [HASH] Loaded existing hash records, {len(hash_data)} records total")
        except json.JSONDecodeError:
            print(f"⚠️ [WARNING] hash.json has invalid format, creating new file")
            hash_data = {}
    
    # Calculate hash for each target file
    updated_files = []
    for file_path in target_files:
        if not os.path.exists(file_path):
            print(f"❌ [ERROR] File does not exist: {file_path}")
            continue
            
        if not os.path.isfile(file_path):
            print(f"❌ [ERROR] Path is not a file: {file_path}")
            continue
            
        # Calculate hash
        try:
            file_hash = calculate_sha256(file_path)
            
            # Update hash data
            hash_data[file_path] = {
                "hash": file_hash,
                "algorithm": "sha256"
            }
            
            updated_files.append(file_path)
            print(f"✅ [UPDATE] {file_path}: {file_hash}")
        except Exception as e:
            print(f"❌ [ERROR] Error calculating hash for {file_path}: {str(e)}")
    
    # Save updated hash data
    if updated_files:
        try:
            with open(hash_file_path, 'w') as f:
                json.dump(hash_data, f, indent=2, ensure_ascii=False)
            print(f"💾 [SAVE] Updated hash values for {len(updated_files)} files to {hash_file_path}")
        except Exception as e:
            print(f"❌ [ERROR] Error saving hash data: {str(e)}")
            return False
        return True
    else:
        print("ℹ️ [INFO] No files were updated")
        return False

def verify_hash_json(target_files):
    """Verify file hashes against hash.json"""
    hash_file_path = "hash.json"
    hash_data = {}
    
    # Try to load existing hash data
    if not os.path.exists(hash_file_path) or os.path.getsize(hash_file_path) == 0:
        print(f"❌ [ERROR] {hash_file_path} does not exist or is empty")
        return False
    
    try:
        with open(hash_file_path, 'r') as f:
            hash_data = json.load(f)
            print(f"📖 [HASH] Loaded hash records, {len(hash_data)} records total")
    except json.JSONDecodeError:
        print(f"❌ [ERROR] {hash_file_path} has invalid format")
        return False
    
    # Verify hash for each target file
    verified_files = []
    failed_files = []
    missing_files = []
    
    for file_path in target_files:
        if not os.path.exists(file_path):
            print(f"❌ [ERROR] File does not exist: {file_path}")
            missing_files.append(file_path)
            continue
            
        if not os.path.isfile(file_path):
            print(f"❌ [ERROR] Path is not a file: {file_path}")
            continue
        
        # Check if file exists in hash data
        if file_path not in hash_data:
            print(f"⚠️ [WARNING] File does not exist in hash records: {file_path}")
            continue
        
        # Calculate current hash
        try:
            current_hash = calculate_sha256(file_path)
            expected_hash = hash_data[file_path]["hash"]
            
            if current_hash == expected_hash:
                verified_files.append(file_path)
                print(f"✅ [VERIFY] {file_path}: {current_hash} - Hash matches")
            else:
                failed_files.append(file_path)
                print(f"❌ [VERIFY] {file_path}: {current_hash} - Hash does not match, expected: {expected_hash}")
        except Exception as e:
            print(f"❌ [ERROR] Error calculating hash for {file_path}: {str(e)}")
    
    # Print summary
    print("\n📊 [VERIFICATION RESULTS]")
    print(f"✅ Verified: {len(verified_files)} files")
    
    if failed_files:
        print(f"❌ Verification failed: {len(failed_files)} files")
        for file in failed_files:
            print(f"  - {file}")
    
    if missing_files:
        print(f"⚠️ Missing files: {len(missing_files)} files")
        for file in missing_files:
            print(f"  - {file}")
    
    return len(failed_files) == 0 and len(missing_files) == 0

def get_target_files(args):
    """Get the list of target files based on the provided arguments"""
    # Get files from hash.json if --all is specified and mode is verify
    if args.all and args.mode == 'verify':
        try:
            with open("hash.json", 'r') as f:
                hash_data = json.load(f)
                target_files = list(hash_data.keys())
                print(f"🔍 [START] Verifying all {len(target_files)} files in hash.json...")
                return target_files
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"❌ [ERROR] Could not load hash.json: {str(e)}")
            sys.exit(1)
    
    # Otherwise process specified files
    if not args.files:
        return None
    
    # Process the files
    target_files = []
    
    for path in args.files:
        if os.path.isfile(path):
            target_files.append(path)
        elif os.path.isdir(path):
            if args.recursive:
                for root, _, files in os.walk(path):
                    for file in files:
                        if args.pattern:
                            import fnmatch
                            if fnmatch.fnmatch(file, args.pattern):
                                target_files.append(os.path.join(root, file))
                        else:
                            target_files.append(os.path.join(root, file))
            else:
                print(f"⚠️ [WARNING] {path} is a directory. Use --recursive option to process directories")
    
    return target_files if target_files else None

def main():
    parser = argparse.ArgumentParser(description='File Hash Management Tool - Update or verify SHA256 hashes of files')
    parser.add_argument('--mode', '-m', choices=['update', 'verify'], required=True, help='Operation mode: update or verify file hashes')
    parser.add_argument('files', nargs='*', help='Paths to files or directories to process')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process directories recursively')
    parser.add_argument('--pattern', '-p', type=str, help='File matching pattern, e.g. "*.py"')
    parser.add_argument('--all', '-a', action='store_true', help='In verify mode, verify all files in hash.json')
    
    args = parser.parse_args()
    
    # Mode-specific validations
    if args.mode == 'update' and args.all:
        print("⚠️ [WARNING] --all option is only valid in verify mode")
        args.all = False
    
    if args.mode == 'verify' and not args.files and not args.all:
        parser.print_help()
        print("\n⚠️ [WARNING] In verify mode, please specify at least one file or directory, or use --all option to verify all files")
        sys.exit(1)
    
    if args.mode == 'update' and not args.files:
        parser.print_help()
        print("\n⚠️ [WARNING] In update mode, please specify at least one file or directory")
        sys.exit(1)
    
    # Get target files
    target_files = get_target_files(args)
    
    if target_files is None:
        if args.mode == 'update':
            print("⚠️ [WARNING] No matching files found")
            sys.exit(1)
        elif args.mode == 'verify' and not args.all:
            print("⚠️ [WARNING] No matching files found")
            sys.exit(1)
    
    # Execute the appropriate function based on the mode
    if args.mode == 'update':
        print(f"🔍 [START] Processing {len(target_files)} files to update hashes...")
        success = update_hash_json(target_files)
        
        if success:
            print("🎉 [COMPLETE] Hash update successful")
            sys.exit(0)
        else:
            print("⚠️ [COMPLETE] Warnings or errors occurred during hash update")
            sys.exit(1)
    
    elif args.mode == 'verify':
        if target_files:
            print(f"🔍 [START] Verifying {len(target_files)} files...")
        
        success = verify_hash_json(target_files)
        
        if success:
            print("🎉 [COMPLETE] All file hash verifications passed")
            sys.exit(0)
        else:
            print("⚠️ [COMPLETE] Hash verification has failures or warnings")
            sys.exit(1)

if __name__ == '__main__':
    main() 