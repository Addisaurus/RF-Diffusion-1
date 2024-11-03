# memory_utils.py
import torch
import gc
from contextlib import contextmanager

@contextmanager
def track_memory():
    """Context manager for tracking GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_allocated = torch.cuda.memory_allocated()
        start_reserved = torch.cuda.memory_reserved()
        
        print(f"\n=== Memory tracking start ===")
        print(f"Allocated: {start_allocated/1024**3:.2f} GB")
        print(f"Reserved:  {start_reserved/1024**3:.2f} GB")
        
        try:
            yield
        finally:
            torch.cuda.synchronize()
            end_allocated = torch.cuda.memory_allocated()
            end_reserved = torch.cuda.memory_reserved()
            
            print(f"\n=== Memory tracking end ===")
            print(f"Allocated: {end_allocated/1024**3:.2f} GB")
            print(f"Reserved:  {end_reserved/1024**3:.2f} GB")
            print(f"Difference allocated: {(end_allocated-start_allocated)/1024**3:.2f} GB")
            print(f"Difference reserved:  {(end_reserved-start_reserved)/1024**3:.2f} GB")

def clear_memory():
    """Helper function to clear memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()