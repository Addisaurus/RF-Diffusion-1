"""
Flexible conditioning system for RF-Diffusion.
Handles creation and validation of conditioning vectors.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class FieldConfig:
    """Configuration for a single conditioning field"""
    type: str  # "categorical" or "continuous"
    values: Optional[List[str]] = None  # For categorical fields
    normalize: bool = False  # For continuous fields
    required: bool = True
    min_value: Optional[float] = None  # For continuous normalization
    max_value: Optional[float] = None  # For continuous normalization

class ConditioningManager:
    """Manages creation and validation of conditioning vectors"""
    
    def __init__(self, config):
        """
        Initialize conditioning manager from config.
        
        Args:
            config: Configuration object containing conditioning settings
        """
        self.enabled_fields = self._get_config_value(config, 'enabled_fields', ['mod_type'])
        raw_field_configs = self._get_config_value(config, 'field_configs', {})
        
        # Convert AttrDict configs to FieldConfig instances
        self.field_configs = {}
        for field, field_config in raw_field_configs.items():
            # Only convert configs for enabled fields
            if field in self.enabled_fields:
                self.field_configs[field] = FieldConfig(
                    type=field_config.get('type'),
                    values=field_config.get('values'),
                    normalize=field_config.get('normalize', False),
                    required=field_config.get('required', True),
                    min_value=field_config.get('min_value'),
                    max_value=field_config.get('max_value')
                )
        
        self._validate_configuration()
        self.conditioning_dim = self._calculate_conditioning_dim()
        
        logger.info(f"Initialized conditioning with fields: {self.enabled_fields}")
        logger.info(f"Total conditioning dimension: {self.conditioning_dim}")

    def _get_default_value(self, config: FieldConfig) -> float:
        """Get appropriate default value for a field configuration"""
        if config.type == "categorical":
            return 0  # Index of default category
        elif config.normalize:
            return 0.0  # Normalized default is 0
        else:
            # For unnormalized continuous, use min_value or 0
            return config.min_value if config.min_value is not None else 0.0

    def _normalize_continuous_value(self, value: float, config: FieldConfig, 
                                  is_default: bool = False) -> float:
        """
        Normalize continuous value with clamping.
        
        Args:
            value: Value to normalize
            config: Field configuration
            is_default: Whether this is a default value for a missing field
        """
        if not config.normalize:
            return value
            
        if is_default:
            return 0.0  # Default values should be normalized to 0
         
        # Clamp value to min/max range
        value = max(config.min_value, min(config.max_value, value))
        
        # Normalize to [0, 1]
        normalized = (value - config.min_value) / (config.max_value - config.min_value)
        return normalized
            
    def _get_config_value(self, config, key: str, default: Any) -> Any:
        """Safely extract value from nested config with default"""
        try:
            return getattr(getattr(config, 'conditioning', None), key, default)
        except AttributeError:
            return default

    def _validate_configuration(self):
        """Validate conditioning configuration"""
        # print("\n=== Validating Configuration ===")
        # print(f"Enabled fields: {self.enabled_fields}")
        # print(f"Field configs: {self.field_configs}")

        # Ensure mod_type is enabled
        if 'mod_type' not in self.enabled_fields:
            raise ValueError("mod_type must be in enabled_fields")
            
        # Validate each field configuration
        for field in self.enabled_fields:
            if field not in self.field_configs:
                if field == 'mod_type':
                    # Set default mod_type configuration if not provided
                    self.field_configs['mod_type'] = FieldConfig(
                        type="categorical",
                        values=['bpsk', 'qpsk', '8psk', 'dqpsk', 
                               '16qam', '64qam', '256qam', 'msk'],
                        required=True
                    )
                else:
                    raise ValueError(f"Missing configuration for enabled field: {field}")
            
            config = self.field_configs[field]
            if not isinstance(config, FieldConfig):
                raise TypeError(f"Field config must be FieldConfig instance: {field}")
                
            if config.type == "categorical" and not config.values:
                raise ValueError(f"Categorical field must specify values: {field}")
            
            if config.type == "continuous" and config.normalize:
                if config.min_value is None or config.max_value is None:
                    raise ValueError(f"Continuous normalized field must specify min/max: {field}")

    def _calculate_conditioning_dim(self) -> int:
        """Calculate total dimension of conditioning vector"""
        dim = 0
        for field in self.enabled_fields:
            config = self.field_configs[field]
            if config.type == "categorical":
                dim += len(config.values)  # One-hot encoding
            else:
                dim += 1  # Single continuous value
        return dim

    def _normalize_continuous_value(self, value: float, config: FieldConfig, is_default: bool = False) -> float:
        """Normalize continuous value with clamping"""
        if not config.normalize:
            return value
        
        if is_default:
            return 0.0

        # Clamp value to min/max range
        value = max(config.min_value, min(config.max_value, value))
        
        # Normalize to [0, 1]
        normalized = (value - config.min_value) / (config.max_value - config.min_value)
        return normalized

    def create_condition_vector(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Create conditioning vector from metadata.
        
        Args:
            metadata: Dictionary containing field values
            
        Returns:
            torch.Tensor: Conditioning vector
        """
        components = []
        
        for field in self.enabled_fields:
            config = self.field_configs[field]
            
            # Check if field is missing
            field_missing = field not in metadata
            
            # Handle missing fields
            if field_missing:
                if config.required:
                    raise ValueError(f"Required field missing from metadata: {field}")
                else:
                    # Use appropriate default value
                    if config.type == "categorical":
                        # For categorical, use one-hot encoding of first value
                        vec = torch.zeros(len(config.values))
                        vec[0] = 1.0  # First value is default
                        components.append(vec)
                    else:
                        # For continuous, use normalized 0
                        components.append(torch.tensor([0.0]))
                    continue
            
            # Process provided values
            value = metadata[field]
            
            if config.type == "categorical":
                if value not in config.values:
                    raise ValueError(f"Invalid value for {field}: {value}")
                # One-hot encode
                vec = torch.zeros(len(config.values))
                idx = config.values.index(value)
                vec[idx] = 1.0
                components.append(vec)
            else:
                # Convert to float and normalize if needed
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    raise ValueError(f"Could not convert {field} value to float: {value}")
                
                if config.normalize:
                    value = self._normalize_continuous_value(value, config, is_default=field_missing)
                components.append(torch.tensor([value]))
        
        # Concatenate all components
        return torch.cat(components)

    def describe_vector(self, vector: torch.Tensor) -> Dict[str, Any]:
        """
        Convert conditioning vector back to human-readable form.
        
        Args:
            vector: Conditioning vector
            
        Returns:
            Dictionary mapping fields to their values
        """
        result = {}
        start_idx = 0
        
        for field in self.enabled_fields:
            config = self.field_configs[field]
            if config.type == "categorical":
                length = len(config.values)
                field_vec = vector[start_idx:start_idx + length]
                if torch.sum(field_vec) > 0:  # If any one-hot encoding
                    idx = torch.argmax(field_vec).item()
                    result[field] = config.values[idx]
                start_idx += length
            else:
                value = vector[start_idx].item()
                if config.normalize:
                    # Denormalize value
                    value = value * (config.max_value - config.min_value) + config.min_value
                result[field] = value
                start_idx += 1
        
        return result