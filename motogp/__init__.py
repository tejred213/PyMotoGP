"""
MotoGP Data Library - Fast, pythonic access to MotoGP data

A data analysis library inspired by FastF1, providing easy access to official
MotoGP qualifying and race data through the PulseLive API.

Example usage:
    >>> import motogp
    >>> session = motogp.load('2025', 'cataluña', 'qualifying')
    >>> df = session.laps.to_dataframe()
    >>> session.plot.lap_times()
"""

__version__ = '0.3.0'
__author__ = 'Tejas Redkar'
__email__ = 'redkartejas213@gmail.com'
__license__ = 'MIT'

from .core.session import Session, load, SessionMetadata
from .core.lap import Lap, LapCollection
from .core.sector import Sector
from .core.rider import Rider, Team
from .api.pulselive import PulseLiveClient, PulseLiveError
from .events import get_event_schedule, update
from .analysis import (
    SessionAnalyzer,
    HistoricalAnalyzer,
    RacePaceEstimator,
)
# RacePaceValidator is intentionally NOT re-exported at top-level.
# It's a research/backtest tool — network-bound, slow, meant for model
# evaluation rather than daily notebook use. Import from the submodule:
#     from motogp.analysis import RacePaceValidator

__all__ = [
    'Session',
    'SessionMetadata',
    'Lap',
    'LapCollection',
    'Sector',
    'Rider',
    'Team',
    'PulseLiveClient',
    'PulseLiveError',
    'SessionAnalyzer',
    'HistoricalAnalyzer',
    'RacePaceEstimator',
    'load',
    'get_event_schedule',
    'update',
]
