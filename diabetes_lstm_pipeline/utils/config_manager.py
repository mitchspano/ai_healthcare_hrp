"""Configuration management system with YAML/JSON support."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation for the diabetes LSTM pipeline."""

    def __init__(self, config_path: Union[str, Path] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file. If None, uses default config.
        """
        self.config_path = (
            Path(config_path) if config_path else Path("configs/default_config.yaml")
        )
        self.config = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, "r") as file:
                if (
                    self.config_path.suffix.lower() == ".yaml"
                    or self.config_path.suffix.lower() == ".yml"
                ):
                    self.config = yaml.safe_load(file)
                elif self.config_path.suffix.lower() == ".json":
                    self.config = json.load(file)
                else:
                    raise ValueError(
                        f"Unsupported configuration file format: {self.config_path.suffix}"
                    )

            logger.info(f"Configuration loaded from {self.config_path}")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key (supports dot notation, e.g., 'data.raw_data_path')
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config_ref = self.config

        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]

        # Set the value
        config_ref[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with a dictionary of values.

        Args:
            updates: Dictionary of configuration updates
        """
        for key, value in updates.items():
            self.set(key, value)

    def save(self, output_path: Union[str, Path] = None) -> None:
        """
        Save current configuration to file.

        Args:
            output_path: Path to save configuration. If None, overwrites original file.
        """
        save_path = Path(output_path) if output_path else self.config_path

        try:
            with open(save_path, "w") as file:
                if save_path.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(self.config, file, default_flow_style=False, indent=2)
                elif save_path.suffix.lower() == ".json":
                    json.dump(self.config, file, indent=2)
                else:
                    raise ValueError(f"Unsupported output format: {save_path.suffix}")

            logger.info(f"Configuration saved to {save_path}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    def validate_required_keys(self, required_keys: list) -> bool:
        """
        Validate that all required configuration keys are present.

        Args:
            required_keys: List of required configuration keys (dot notation supported)

        Returns:
            True if all required keys are present, False otherwise
        """
        missing_keys = []

        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)

        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False

        return True

    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()

    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ConfigManager(config_path={self.config_path}, keys={list(self.config.keys())})"

    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return self.__str__()


# Global configuration instance
_global_config = None


def get_config(config_path: Union[str, Path] = None) -> ConfigManager:
    """
    Get the global configuration instance.

    Args:
        config_path: Path to configuration file (only used on first call)

    Returns:
        Global ConfigManager instance
    """
    global _global_config

    if _global_config is None:
        _global_config = ConfigManager(config_path)

    return _global_config


def reload_config(config_path: Union[str, Path] = None) -> ConfigManager:
    """
    Reload the global configuration instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Reloaded ConfigManager instance
    """
    global _global_config
    _global_config = ConfigManager(config_path)
    return _global_config
