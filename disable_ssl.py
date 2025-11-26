#!/usr/bin/env python3
"""
Utility script to disable SSL in austin-to-santa.py for easier testing.
This is useful when using Cloudflare Tunnel or other services that provide SSL.
"""

import re
import sys
import os

def disable_ssl():
    filename = 'austin-to-santa.py'

    if not os.path.exists(filename):
        print(f"❌ Error: {filename} not found in current directory")
        return False

    # Read the file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to find the uvicorn.run line with SSL
    ssl_pattern = r"uvicorn\.run\(app,\s*host='0\.0\.0\.0',\s*port=7777,\s*ssl_certfile='[^']+',\s*ssl_keyfile='[^']+'\)"
    no_ssl_replacement = "uvicorn.run(app, host='0.0.0.0', port=7777)"

    # Check if SSL is currently enabled
    if re.search(ssl_pattern, content):
        # Replace with no SSL version
        new_content = re.sub(ssl_pattern, no_ssl_replacement, content)

        # Backup the original file
        backup_name = f"{filename}.backup"
        with open(backup_name, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created backup: {backup_name}")

        # Write the modified content
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ SSL disabled in {filename}")
        print("   The server will now run on HTTP (port 7777)")
        print("   Perfect for use with Cloudflare Tunnel or ngrok!")
        return True

    elif "uvicorn.run(app, host='0.0.0.0', port=7777)" in content:
        print("ℹ️  SSL is already disabled")
        return True
    else:
        print(f"⚠️  Warning: Could not find the uvicorn.run line in {filename}")
        print("   Please check the file manually")
        return False

def enable_ssl():
    filename = 'austin-to-santa.py'

    if not os.path.exists(filename):
        print(f"❌ Error: {filename} not found in current directory")
        return False

    # Read the file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to find the uvicorn.run line without SSL
    no_ssl_pattern = r"uvicorn\.run\(app,\s*host='0\.0\.0\.0',\s*port=7777\)"
    ssl_replacement = "uvicorn.run(app, host='0.0.0.0', port=7777, ssl_certfile='static/sec/cert.pem', ssl_keyfile='static/sec/privkey.pem')"

    # Check if SSL is currently disabled
    if re.search(no_ssl_pattern, content) and 'ssl_certfile' not in content:
        # Replace with SSL version
        new_content = re.sub(no_ssl_pattern, ssl_replacement, content)

        # Write the modified content
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ SSL enabled in {filename}")
        print("   The server will now run on HTTPS (port 7777)")
        print("   Make sure you have certificates in static/sec/")
        return True

    elif "ssl_certfile" in content:
        print("ℹ️  SSL is already enabled")
        return True
    else:
        print(f"⚠️  Warning: Could not find the uvicorn.run line in {filename}")
        print("   Please check the file manually")
        return False

if __name__ == "__main__":
    print("🎅 Santa Claus is Calling - SSL Configuration Tool")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == '--enable':
        enable_ssl()
    else:
        disable_ssl()
        print("\nTip: Use 'python disable_ssl.py --enable' to re-enable SSL")