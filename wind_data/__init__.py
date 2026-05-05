"""
Wind Data Module
Handles wind field representation and data access.
"""

from .wind_field import WindField
from .openfoam_loader import extract_openfoam_wind

__all__ = ['WindField', 'extract_openfoam_wind']
