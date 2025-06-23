"""
MorganVuoksi Elite Terminal - UI Module
Bloomberg-style quantitative finance terminal user interface components.
"""

__version__ = "2.0.0"
__author__ = "MorganVuoksi Team"

# Try to import components, but don't fail if they're not available
try:
    from .utils.theme import BloombergTheme
    from .utils.session import SessionManager
    from .utils.cache import CacheManager, cache_get, cache_set, cache_clear
    
    __all__ = [
        'BloombergTheme',
        'SessionManager', 
        'CacheManager',
        'cache_get',
        'cache_set',
        'cache_clear'
    ]
    
except ImportError:
    # Graceful fallback if imports fail
    __all__ = []
    
    # Create dummy classes for development
    class BloombergTheme:
        @staticmethod
        def apply_theme(): pass
        @staticmethod
        def create_metric_card(label, value, change=None, change_type='neutral'):
            return f"<div><strong>{label}:</strong> {value} {change or ''}</div>"
        @staticmethod
        def create_header(title, status='live'):
            return f"<h3>{title}</h3>"
    
    class SessionManager:
        @staticmethod
        def initialize(): pass
    
    class CacheManager:
        pass
    
    def cache_get(*args, **kwargs): return None
    def cache_set(*args, **kwargs): pass
    def cache_clear(*args, **kwargs): pass