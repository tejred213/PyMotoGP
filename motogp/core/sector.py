"""Sector module - Sector-level lap analysis"""

from typing import Optional
from dataclasses import dataclass
from datetime import timedelta


@dataclass
class Sector:
    """
    Represents a single sector within a lap.
    
    A lap is typically divided into 3 sectors for detailed timing analysis.
    
    Attributes:
        sector_number: Sector ID (1, 2, or 3)
        time: Sector elapsed time
        speed: Top speed in sector
        delta_to_best: Time delta vs. best sector time
        is_valid: Whether this sector is valid
    """
    sector_number: int
    time: timedelta
    speed: float
    delta_to_best: Optional[timedelta] = None
    is_valid: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'sector_number': self.sector_number,
            'time_ms': self.time.total_seconds() * 1000,
            'speed': self.speed,
            'delta_to_best_ms': (
                self.delta_to_best.total_seconds() * 1000
                if self.delta_to_best else None
            ),
            'is_valid': self.is_valid,
        }
    
    def __repr__(self) -> str:
        ms = int(self.time.total_seconds() * 1000)
        s = ms // 1000
        m = ms % 1000
        return f"Sector({self.sector_number}, {s}.{m:03d}s, {self.speed}km/h)"
