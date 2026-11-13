# LoggerMixin

A Python mixin class that provides standardized logging functionality with colored output, automatic logger naming, and configurable formatting for any class that inherits from it.

## Overview

The LoggerMixin provides a consistent logging interface across your project by automatically creating appropriately named loggers with colored console output. Classes that inherit from this mixin gain access to a `logger` property that creates formatted, colored log messages with contextual information including timestamps, log levels, class names, and function names.

## Features

- **Automatic Logger Naming**: Logger names automatically generated from class module and name
- **Colored Output**: Different colors for different log levels using colorama
- **Rich Formatting**: Includes timestamps, log levels, class names, function names, and messages
- **Configurable Log Levels**: Class-level and instance-level log level configuration
- **Zero Configuration**: Works immediately upon inheritance with sensible defaults
- **Thread Safe**: Safe for use in multi-threaded applications

## Installation

### Dependencies
```bash
pip install colorama
```

The mixin only requires Python's built-in `logging` module and the `colorama` package for colored output.

## Quick Start

### Basic Usage
```python
from logger_config import LoggerMixin

class MyClass(LoggerMixin):
    def __init__(self):
        self.logger.info("MyClass initialized")
    
    def do_something(self):
        self.logger.debug("Starting operation")
        try:
            # Your code here
            result = 42
            self.logger.info(f"Operation completed with result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Operation failed: {e}")
            raise

# Usage
obj = MyClass()
obj.do_something()
```

**Output:**
```
14:23:45 | INFO     | mymodule.MyClass | __init__        | MyClass initialized
14:23:45 | DEBUG    | mymodule.MyClass | do_something    | Starting operation
14:23:45 | INFO     | mymodule.MyClass | do_something    | Operation completed with result: 42
```

### Configuring Log Levels
```python
class MyClass(LoggerMixin):
    def __init__(self, debug=False):
        # Configure log level at instance level
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        self.logger.debug("Debug mode enabled")
        self.logger.info("MyClass ready")

# Usage
debug_obj = MyClass(debug=True)  # Will show debug messages
normal_obj = MyClass(debug=False)  # Only shows info and above
```

## Color Scheme

The mixin provides color-coded log levels for easy visual scanning:

| Log Level | Color | Description |
|-----------|-------|-------------|
| DEBUG | Green | Development and troubleshooting information |
| INFO | Green | General operational messages |
| WARNING | Yellow | Warnings that don't stop execution |
| ERROR | Red | Error conditions |
| CRITICAL | Bright Red | Critical errors requiring immediate attention |

## Log Format

The default log format includes:
```
%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)-15s | %(message)s
```

**Example output:**
```
14:23:45 | INFO     | shipshape.ShipShapeBasic | _validate_inputs | Input validation passed
14:23:45 | WARNING  | shipshape.ShipShapeBasic | _calculate_C_M   | C_M calculation resulted in value outside normal range
14:23:45 | ERROR    | shipshape.ShipShapeBasic | _load_data       | Failed to load ship database: File not found
```

## Advanced Usage

### Multiple Inheritance
```python
from logger_config import LoggerMixin

class DataProcessor:
    def process_data(self, data):
        return data * 2

class MyAdvancedClass(DataProcessor, LoggerMixin):
    def __init__(self):
        super().__init__()
        self.logger.info("Advanced class with multiple inheritance")
    
    def complex_operation(self, data):
        self.logger.debug(f"Processing data: {data}")
        result = self.process_data(data)
        self.logger.info(f"Processing complete: {result}")
        return result
```

### Context-Aware Logging
```python
class DatabaseManager(LoggerMixin):
    def __init__(self, db_name):
        self.db_name = db_name
        self.logger.info(f"Database manager initialized for: {db_name}")
    
    def connect(self):
        try:
            self.logger.debug("Attempting database connection")
            # Connection logic here
            self.logger.info(f"Successfully connected to {self.db_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.db_name}: {e}")
            raise
    
    def query(self, sql):
        self.logger.debug(f"Executing query: {sql[:50]}...")
        # Query logic here
        self.logger.info("Query executed successfully")
```

### Error Handling and Debugging
```python
class ShipCalculator(LoggerMixin):
    def __init__(self):
        self.logger.info("Ship calculator initialized")
    
    def calculate_displacement(self, length, beam, draft, cb):
        self.logger.debug(f"Calculating displacement: L={length}, B={beam}, T={draft}, Cb={cb}")
        
        try:
            if any(val <= 0 for val in [length, beam, draft, cb]):
                self.logger.warning("Negative or zero values detected in displacement calculation")
            
            displacement = length * beam * draft * cb * 1.025  # seawater density
            
            if displacement > 500000:  # Very large ship
                self.logger.warning(f"Unusually large displacement calculated: {displacement:.0f} tonnes")
            
            self.logger.info(f"Displacement calculated: {displacement:.2f} tonnes")
            return displacement
            
        except Exception as e:
            self.logger.error(f"Displacement calculation failed: {e}")
            self.logger.debug(f"Parameters were: L={length}, B={beam}, T={draft}, Cb={cb}")
            raise
```

## Creating Your Own Compatible Mixin

If you need to create your own logging mixin that maintains compatibility with classes using the existing LoggerMixin, follow this interface:

### Required Interface
Your custom mixin must provide:
1. A `logger` property that returns a `logging.Logger` instance
2. The logger should be named using the class's module and name
3. The logger should be properly configured for immediate use

### Compatible Implementation Example
```python
import logging
from typing import Optional

class CustomLoggerMixin:
    """
    Custom logger mixin compatible with existing LoggerMixin usage
    """
    
    # Class-level configuration
    DEFAULT_LOG_LEVEL = logging.INFO
    CUSTOM_FORMAT = '%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger: Optional[logging.Logger] = None
    
    @property
    def logger(self) -> logging.Logger:
        """
        Returns a configured logger for this class instance.
        Must maintain compatibility with existing LoggerMixin interface.
        """
        if self._logger is None:
            # Create logger name from class module and name (required for compatibility)
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = logging.getLogger(logger_name)
            
            # Only configure if not already configured
            if not self._logger.handlers:
                self._configure_logger(self._logger)
        
        return self._logger
    
    def _configure_logger(self, logger: logging.Logger) -> None:
        """Configure the logger with custom settings"""
        # Create handler
        handler = logging.StreamHandler()
        
        # Create formatter (customize as needed)
        formatter = logging.Formatter(
            fmt=self.CUSTOM_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Apply configuration
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(self.DEFAULT_LOG_LEVEL)
        
        # Optional: Add custom behavior
        logger.info(f"Custom logger initialized for {logger.name}")

# Usage example - drop-in replacement
class ExampleClass(CustomLoggerMixin):
    def __init__(self):
        super().__init__()
        self.logger.info("This works with the custom mixin")
    
    def test_method(self):
        self.logger.debug("Debug message")
        self.logger.warning("Warning message")
```

### Advanced Custom Mixin with Additional Features
```python
import logging
import json
from datetime import datetime
from pathlib import Path

class AdvancedLoggerMixin:
    """
    Advanced logger mixin with file output and JSON formatting options
    """
    
    def __init__(self, *args, log_to_file=False, log_file_path=None, json_logs=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None
        self._log_to_file = log_to_file
        self._log_file_path = log_file_path
        self._json_logs = json_logs
    
    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            # Maintain compatibility: use module.class naming
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = logging.getLogger(logger_name)
            
            if not self._logger.handlers:
                self._setup_handlers(self._logger)
        
        return self._logger
    
    def _setup_handlers(self, logger: logging.Logger) -> None:
        """Setup console and optional file handlers"""
        
        # Console handler (for compatibility)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)-15s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Optional file handler
        if self._log_to_file:
            file_path = self._log_file_path or f"logs/{logger.name}.log"
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(file_path)
            
            if self._json_logs:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s'
                )
                file_handler.setFormatter(file_formatter)
            
            logger.addHandler(file_handler)
        
        logger.setLevel(logging.INFO)

class JSONFormatter(logging.Formatter):
    """Custom formatter for JSON log output"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'function': record.funcName,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# Usage with advanced features
class ProductionClass(AdvancedLoggerMixin):
    def __init__(self):
        super().__init__(
            log_to_file=True,
            log_file_path='production_logs/app.log',
            json_logs=True
        )
        self.logger.info("Production class initialized with file logging")
```

## Compatibility Requirements

When creating your own logger mixin, ensure compatibility by:

### 1. Property Interface
```python
@property
def logger(self) -> logging.Logger:
    # Must return a configured logging.Logger instance
    pass
```

### 2. Logger Naming Convention
```python
logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
logger = logging.getLogger(logger_name)
```

### 3. Immediate Usability
The logger must be ready to use immediately after accessing the property, without requiring additional configuration calls.

### 4. Handler Management
Avoid duplicate handlers by checking if handlers already exist:
```python
if not logger.handlers:
    # Configure handlers
    pass
```

## Testing Your Custom Mixin

```python
def test_logger_compatibility():
    """Test that your custom mixin works with existing code"""
    
    class TestClass(YourCustomLoggerMixin):
        def __init__(self):
            super().__init__()
            self.logger.info("Test initialization")
        
        def test_method(self):
            self.logger.debug("Debug test")
            self.logger.warning("Warning test")
            self.logger.error("Error test")
    
    # Should work without any additional setup
    test_obj = TestClass()
    test_obj.test_method()
    
    # Verify logger properties
    assert hasattr(test_obj, 'logger')
    assert isinstance(test_obj.logger, logging.Logger)
    assert test_obj.logger.name.endswith('TestClass')
    
    print("Compatibility test passed!")

if __name__ == "__main__":
    test_logger_compatibility()
```

## Best Practices

### For Using LoggerMixin
1. **Always call super().__init__()** when overriding `__init__` in classes that inherit from LoggerMixin
2. **Use appropriate log levels**: DEBUG for development, INFO for normal operations, WARNING for issues that don't stop execution, ERROR for failures
3. **Include context in messages**: Use f-strings to include relevant variable values
4. **Log at function entry/exit for complex operations** to aid debugging

### For Creating Custom Mixins
1. **Maintain the property interface** - other code expects `self.logger` to work
2. **Use the standard naming convention** for logger names
3. **Avoid breaking existing functionality** - test with existing classes
4. **Document any additional features** or configuration options
5. **Handle edge cases** like multiple inheritance gracefully

## Examples in Different Contexts

### Web Application
```python
class WebAPIHandler(LoggerMixin):
    def handle_request(self, request):
        self.logger.info(f"Received {request.method} request to {request.path}")
        try:
            response = self.process_request(request)
            self.logger.info(f"Request completed successfully: {response.status_code}")
            return response
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise
```

### Data Processing
```python
class DataPipeline(LoggerMixin):
    def __init__(self, pipeline_name):
        self.pipeline_name = pipeline_name
        self.logger.info(f"Data pipeline '{pipeline_name}' initialized")
    
    def process_batch(self, data_batch):
        self.logger.info(f"Processing batch of {len(data_batch)} records")
        processed = 0
        errors = 0
        
        for record in data_batch:
            try:
                self.process_record(record)
                processed += 1
            except Exception as e:
                errors += 1
                self.logger.warning(f"Record processing failed: {e}")
        
        self.logger.info(f"Batch complete: {processed} processed, {errors} errors")
```

### Scientific Computing
```python
class ShipHullAnalyzer(LoggerMixin):
    def __init__(self):
        self.logger.info("Hull analyzer initialized")
    
    def analyze_resistance(self, hull_params):
        self.logger.debug(f"Starting resistance analysis with params: {hull_params}")
        
        try:
            resistance = self.calculate_resistance(hull_params)
            
            if resistance > 1000:  # kN
                self.logger.warning(f"High resistance calculated: {resistance:.2f} kN")
            
            self.logger.info(f"Resistance analysis complete: {resistance:.2f} kN")
            return resistance
            
        except Exception as e:
            self.logger.error(f"Resistance analysis failed: {e}")
            raise
```

## Troubleshooting

### Common Issues

**Logger not showing colors**
- Ensure `colorama` is installed: `pip install colorama`
- Check if your terminal supports ANSI color codes

**Duplicate log messages**
- Multiple instances may be creating duplicate handlers
- Check that `if not logger.handlers:` condition is working properly

**Log level not changing**
- Set log level after accessing the logger: `self.logger.setLevel(logging.DEBUG)`
- Check if parent loggers have different levels set

**Custom mixin not working with existing classes**
- Verify the `logger` property returns a `logging.Logger` instance
- Ensure logger naming follows the `module.classname` convention
- Test with a simple existing class that uses LoggerMixin

## License and Contribution

This LoggerMixin is designed for project-wide consistency in logging. When extending or modifying:

1. Maintain backward compatibility with existing classes
2. Test thoroughly with existing codebase
3. Document any new features or configuration options
4. Follow the established patterns for logger naming and configuration

## Authors

- **Ben Noble** - TU Delft
- Project: T1.6 Deliverable - Digital Twin for Green Shipping