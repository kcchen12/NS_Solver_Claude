"""
Configuration file parser for the NS_Solver.

Supports reading configuration from a text file with key=value pairs.
Lines starting with '#' are treated as comments.

Usage:
    config = ConfigParser('config.txt')
    nx = config.get('nx', default=64, dtype=int)
    re = config.get('re', default=100.0, dtype=float)
    cylinder = config.get('cylinder', default=False, dtype=bool)
"""

import os
from typing import Any, Dict


_BOOL_TRUE = {"true", "1", "yes", "on"}
_BOOL_FALSE = {"false", "0", "no", "off"}


class ConfigParser:
    """Simple configuration file parser for NS_Solver settings."""

    def __init__(self, filepath: str = 'config.txt'):
        """
        Initialize parser and read configuration file.

        Parameters
        ----------
        filepath : str
            Path to configuration file.
        """
        self.filepath = filepath
        self.config: Dict[str, str] = {}
        self.read()

    def read(self) -> None:
        """Read and parse configuration file."""
        if not os.path.exists(self.filepath):
            print(f"Warning: Configuration file '{self.filepath}' not found. "
                  "Using default values.")
            return

        try:
            with open(self.filepath, 'r') as f:
                for line in f:
                    line = line.split("#", 1)[0].strip()
                    # Skip empty lines and comments
                    if not line:
                        continue
                    # Parse key = value
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        self.config[key] = value
        except IOError as e:
            print(f"Error reading config file '{self.filepath}': {e}")

    def get(self, key: str, default: Any = None, dtype: type = str) -> Any:
        """
        Get a configuration value, with optional type conversion.

        Parameters
        ----------
        key : str
            Configuration key to retrieve.
        default : Any, optional
            Default value if key not found.
        dtype : type, optional
            Data type to convert value to. Options: int, float, bool, str.

        Returns
        -------
        value
            Configuration value converted to specified type, or default.
        """
        if key not in self.config:
            return default

        raw_value = self.config[key]
        value_str = raw_value.strip().lower()

        # Handle boolean conversion
        if dtype == bool:
            if value_str in _BOOL_TRUE:
                return True
            elif value_str in _BOOL_FALSE:
                return False
            else:
                return default

        # Handle numeric and string types
        try:
            if dtype == int:
                return int(raw_value)
            elif dtype == float:
                return float(raw_value)
            else:
                return raw_value
        except (ValueError, TypeError):
            print(f"Warning: Could not convert '{key}={raw_value}' "
                  f"to {dtype.__name__}. Using default value {default}.")
            return default

    def get_all(self) -> Dict[str, str]:
        """Return all configuration key-value pairs."""
        return self.config.copy()

    def __repr__(self) -> str:
        return f"ConfigParser(filepath='{self.filepath}', keys={list(self.config.keys())})"
