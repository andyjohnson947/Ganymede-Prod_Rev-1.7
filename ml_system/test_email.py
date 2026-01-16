#!/usr/bin/env python3
"""
Test Email Functionality
Quick test to verify email config and sending works
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_email():
    """Test email sending"""
    print("=" * 60)
    print("EMAIL CONFIGURATION TEST")
    print("=" * 60)

    # Load email config
    email_config_file = project_root / 'config' / 'email_config.json'
    print(f"\n1. Looking for config at: {email_config_file}")

    if not email_config_file.exists():
        print("   ❌ Email config file not found!")
        return False

    print("   ✅ Config file found")

    with open(email_config_file, 'r') as f:
        config = json.load(f)

    print(f"\n2. Email Config:")
    print(f"   Enabled: {config.get('enabled')}")
    print(f"   From: {config.get('from_email')}")
    print(f"   To: {config.get('to_email')}")
    print(f"   SMTP Server: {config.get('smtp_server')}")
    print(f"   SMTP Port: {config.get('smtp_port')}")
    print(f"   Password: {'*' * len(config.get('password', ''))}")

    if not config.get('enabled'):
        print("\n   ❌ Email is disabled in config!")
        print("   Set 'enabled': true in config/email_config.json")
        return False

    print("\n3. Testing SMTP connection...")

    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        from datetime import datetime

        # Create test message
        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = config['to_email']
        msg['Subject'] = f"ML System Test Email - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        test_body = f"""
This is a test email from your Ganymede Trading Bot ML System.

If you're receiving this, email notifications are working correctly!

Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
From: ML System Test Script
"""

        msg.attach(MIMEText(test_body, 'plain'))

        # Connect and send
        print(f"   Connecting to {config['smtp_server']}:{config['smtp_port']}...")
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            print("   Starting TLS...")
            server.starttls()

            print(f"   Logging in as {config['from_email']}...")
            server.login(config['from_email'], config['password'])

            print(f"   Sending email to {config['to_email']}...")
            server.send_message(msg)

        print("\n" + "=" * 60)
        print("✅ SUCCESS! Test email sent successfully!")
        print("=" * 60)
        print(f"\nCheck your inbox at: {config['to_email']}")
        print("(May take a few seconds to arrive)")
        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ FAILED! Email test failed")
        print("=" * 60)
        print(f"\nError: {e}")
        print("\nPossible issues:")
        print("1. Incorrect SMTP server or port")
        print("2. Invalid email/password")
        print("3. App-specific password not set (for Gmail)")
        print("4. 2FA not enabled (for Gmail)")
        print("5. Network/firewall blocking SMTP")
        return False

if __name__ == '__main__':
    test_email()
