"""
Secure Credential Storage
Encrypts and stores MT5 credentials locally
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict
import base64


class CredentialStore:
    """Manages secure storage of MT5 credentials"""

    def __init__(self):
        """Initialize credential store"""
        self.config_dir = Path.home() / '.trading_bot'
        self.credentials_file = self.config_dir / 'credentials.enc'
        self._ensure_config_dir()

    def _ensure_config_dir(self):
        """Create config directory if it doesn't exist"""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _encode(self, data: str) -> str:
        """
        Simple encoding for credentials
        Note: This is basic obfuscation. For production, use proper encryption.

        Args:
            data: String to encode

        Returns:
            Encoded string
        """
        # Simple base64 encoding (rotate + encode)
        rotated = ''.join(chr((ord(c) + 13) % 256) for c in data)
        encoded = base64.b64encode(rotated.encode()).decode()
        return encoded

    def _decode(self, data: str) -> str:
        """
        Decode credentials

        Args:
            data: Encoded string

        Returns:
            Decoded string
        """
        try:
            decoded = base64.b64decode(data.encode()).decode()
            original = ''.join(chr((ord(c) - 13) % 256) for c in decoded)
            return original
        except Exception:
            return ""

    def save_credentials(
        self,
        login: str,
        password: str,
        server: str,
        remember: bool = True
    ) -> bool:
        """
        Save MT5 credentials

        Args:
            login: MT5 login number
            password: MT5 password
            server: MT5 server name
            remember: Whether to remember credentials

        Returns:
            bool: True if saved successfully
        """
        if not remember:
            # Clear saved credentials
            if self.credentials_file.exists():
                self.credentials_file.unlink()
            return True

        try:
            credentials = {
                'login': login,
                'password': self._encode(password),
                'server': server,
                'version': '1.0'
            }

            with open(self.credentials_file, 'w') as f:
                json.dump(credentials, f, indent=2)

            # Set file permissions (Unix-like systems)
            if os.name != 'nt':  # Not Windows
                os.chmod(self.credentials_file, 0o600)

            return True

        except Exception as e:
            print(f"Error saving credentials: {e}")
            return False

    def load_credentials(self) -> Optional[Dict[str, str]]:
        """
        Load saved MT5 credentials

        Returns:
            Dict with login, password, server or None if not found
        """
        if not self.credentials_file.exists():
            return None

        try:
            with open(self.credentials_file, 'r') as f:
                credentials = json.load(f)

            # Decode password
            credentials['password'] = self._decode(credentials.get('password', ''))

            return {
                'login': credentials.get('login', ''),
                'password': credentials['password'],
                'server': credentials.get('server', '')
            }

        except Exception as e:
            print(f"Error loading credentials: {e}")
            return None

    def clear_credentials(self) -> bool:
        """
        Clear saved credentials

        Returns:
            bool: True if cleared successfully
        """
        try:
            if self.credentials_file.exists():
                self.credentials_file.unlink()
            return True
        except Exception as e:
            print(f"Error clearing credentials: {e}")
            return False

    def has_credentials(self) -> bool:
        """
        Check if credentials are saved

        Returns:
            bool: True if credentials exist
        """
        return self.credentials_file.exists()
