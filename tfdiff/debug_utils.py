# debug_utils.py
import logging

# Create module-specific loggers
shape_logger = logging.getLogger('shape_debug')
value_logger = logging.getLogger('value_debug')

def configure_logging(params):
    """Configure logging based on params"""
    # Base configuration
    logging.basicConfig(
        #format='%(name)s - %(levelname)s - %(message)s',
        format='%(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    # Get debug config with defaults
    debug_config = getattr(params, 'debug_config', {})
    if debug_config is None:
        debug_config = {}
    
    # Configure shape debugging
    shape_level = logging.DEBUG if debug_config.get('shape_debug', False) else logging.INFO
    shape_logger.setLevel(shape_level)
    
    # Configure value debugging
    value_level = logging.DEBUG if debug_config.get('value_debug', False) else logging.INFO 
    value_logger.setLevel(value_level)
    
    # Configure module-specific levels if needed
    debug_modules = debug_config.get('debug_modules', {})
    if debug_modules:
        for module, enabled in debug_modules.items():
            logger = logging.getLogger(module)
            logger.setLevel(logging.DEBUG if enabled else logging.INFO)
    
    # Set overall log level
    log_level = debug_config.get('log_level', 'INFO')
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))