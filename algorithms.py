"""
Page replacement algorithms implementation for virtual memory simulation.
"""

from collections import deque, OrderedDict
from typing import List, Tuple, Dict, Any
import copy

class PageReplacementAlgorithm:
    """Base class for page replacement algorithms."""
    
    def __init__(self, frame_count: int):
        self.frame_count = frame_count
        self.frames = []
        self.page_faults = 0
        self.page_hits = 0
        self.access_history = []
        self.frame_history = []
        
    def reset(self):
        """Reset the algorithm state."""
        self.frames = []
        self.page_faults = 0
        self.page_hits = 0
        self.access_history = []
        self.frame_history = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get algorithm statistics."""
        total_accesses = self.page_faults + self.page_hits
        hit_ratio = (self.page_hits / total_accesses) if total_accesses > 0 else 0
        fault_ratio = (self.page_faults / total_accesses) if total_accesses > 0 else 0
        
        return {
            'page_faults': self.page_faults,
            'page_hits': self.page_hits,
            'total_accesses': total_accesses,
            'hit_ratio': hit_ratio,
            'fault_ratio': fault_ratio,
            'access_latency': self.calculate_access_latency()
        }
    
    def calculate_access_latency(self) -> float:
        """Calculate average access latency (page faults have higher latency)."""
        # Assume page hit = 1 time unit, page fault = 10 time units
        if self.page_faults + self.page_hits == 0:
            return 0
        return (self.page_hits * 1 + self.page_faults * 10) / (self.page_hits + self.page_faults)

class FIFOAlgorithm(PageReplacementAlgorithm):
    """First In First Out page replacement algorithm."""
    
    def __init__(self, frame_count: int):
        super().__init__(frame_count)
        self.queue = deque()
    
    def reset(self):
        super().reset()
        self.queue = deque()
    
    def access_page(self, page: int) -> Tuple[bool, List[int]]:
        """
        Access a page and return (is_fault, current_frames).
        """
        is_fault = page not in self.frames
        
        if is_fault:
            self.page_faults += 1
            
            if len(self.frames) < self.frame_count:
                # Frame available
                self.frames.append(page)
                self.queue.append(page)
            else:
                # Replace oldest page
                oldest_page = self.queue.popleft()
                oldest_index = self.frames.index(oldest_page)
                self.frames[oldest_index] = page
                self.queue.append(page)
        else:
            self.page_hits += 1
        
        # Record access
        self.access_history.append({
            'page': page,
            'is_fault': is_fault,
            'frames': copy.deepcopy(self.frames)
        })
        
        return is_fault, copy.deepcopy(self.frames)

class LRUAlgorithm(PageReplacementAlgorithm):
    """Least Recently Used page replacement algorithm."""
    
    def __init__(self, frame_count: int):
        super().__init__(frame_count)
        self.usage_order = OrderedDict()
    
    def reset(self):
        super().reset()
        self.usage_order = OrderedDict()
    
    def access_page(self, page: int) -> Tuple[bool, List[int]]:
        """
        Access a page and return (is_fault, current_frames).
        """
        is_fault = page not in self.frames
        
        if is_fault:
            self.page_faults += 1
            
            if len(self.frames) < self.frame_count:
                # Frame available
                self.frames.append(page)
            else:
                # Replace least recently used page
                lru_page = next(iter(self.usage_order))
                lru_index = self.frames.index(lru_page)
                self.frames[lru_index] = page
                del self.usage_order[lru_page]
            
            self.usage_order[page] = True
        else:
            self.page_hits += 1
            # Update usage order
            del self.usage_order[page]
            self.usage_order[page] = True
        
        # Record access
        self.access_history.append({
            'page': page,
            'is_fault': is_fault,
            'frames': copy.deepcopy(self.frames)
        })
        
        return is_fault, copy.deepcopy(self.frames)

class OptimalAlgorithm(PageReplacementAlgorithm):
    """Optimal page replacement algorithm (Belady's algorithm)."""
    
    def __init__(self, frame_count: int, reference_string: List[int]):
        super().__init__(frame_count)
        self.reference_string = reference_string
        self.current_index = 0
    
    def reset(self):
        super().reset()
        self.current_index = 0
    
    def access_page(self, page: int) -> Tuple[bool, List[int]]:
        """
        Access a page and return (is_fault, current_frames).
        """
        is_fault = page not in self.frames
        
        if is_fault:
            self.page_faults += 1
            
            if len(self.frames) < self.frame_count:
                # Frame available
                self.frames.append(page)
            else:
                # Replace page that will be used farthest in future
                farthest_page = self._find_farthest_page()
                farthest_index = self.frames.index(farthest_page)
                self.frames[farthest_index] = page
        else:
            self.page_hits += 1
        
        # Record access
        self.access_history.append({
            'page': page,
            'is_fault': is_fault,
            'frames': copy.deepcopy(self.frames)
        })
        
        self.current_index += 1
        return is_fault, copy.deepcopy(self.frames)
    
    def _find_farthest_page(self) -> int:
        """Find the page that will be used farthest in the future."""
        farthest_distance = -1
        farthest_page = self.frames[0]
        
        for frame_page in self.frames:
            # Find next occurrence of this page
            next_use = float('inf')
            for i in range(self.current_index + 1, len(self.reference_string)):
                if self.reference_string[i] == frame_page:
                    next_use = i
                    break
            
            if next_use > farthest_distance:
                farthest_distance = next_use
                farthest_page = frame_page
        
        return farthest_page

def simulate_algorithm(algorithm_class, frame_count: int, reference_string: List[int]) -> Tuple[PageReplacementAlgorithm, List[Dict]]:
    """
    Simulate a page replacement algorithm with given parameters.
    Returns the algorithm instance and step-by-step history.
    """
    if algorithm_class == OptimalAlgorithm:
        algorithm = algorithm_class(frame_count, reference_string)
    else:
        algorithm = algorithm_class(frame_count)
    
    algorithm.reset()
    
    for page in reference_string:
        algorithm.access_page(page)
    
    return algorithm, algorithm.access_history
