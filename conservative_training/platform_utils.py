"""Platform helpers only; they do not alter the scientific computation."""
import sys


def max_rss_bytes():
    try:
        import resource
    except ImportError:
        try:
            import psutil
            return getattr(psutil.Process().memory_info(), 'peak_wset', None)
        except ImportError:
            return None
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)
