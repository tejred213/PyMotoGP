"""Tests for the core session module"""

import pytest
from datetime import timedelta
from motogp.core.session import Session
from motogp.core.lap import Lap, LapCollection
from motogp.core.rider import Rider, Team


@pytest.fixture
def sample_laps() -> list[Lap]:
    """Create sample lap data for testing"""
    return [
        Lap(
            lap_number=1,
            rider_name='Bagnaia',
            rider_number=1,
            lap_time=timedelta(seconds=99, milliseconds=123),
            sector_times=[
                timedelta(seconds=32, milliseconds=100),
                timedelta(seconds=33, milliseconds=500),
                timedelta(seconds=33, milliseconds=523),
            ],
            top_speed=320.5,
            tyre_compound='soft',
        ),
        Lap(
            lap_number=2,
            rider_name='Bagnaia',
            rider_number=1,
            lap_time=timedelta(seconds=98, milliseconds=950),
            sector_times=[
                timedelta(seconds=32, milliseconds=50),
                timedelta(seconds=33, milliseconds=400),
                timedelta(seconds=33, milliseconds=500),
            ],
            top_speed=321.0,
            tyre_compound='soft',
        ),
        Lap(
            lap_number=1,
            rider_name='Márquez',
            rider_number=93,
            lap_time=timedelta(seconds=99, milliseconds=500),
            sector_times=[
                timedelta(seconds=32, milliseconds=200),
                timedelta(seconds=33, milliseconds=600),
                timedelta(seconds=33, milliseconds=700),
            ],
            top_speed=319.5,
            tyre_compound='soft',
        ),
    ]


class TestLapCollection:
    """Tests for LapCollection"""
    
    def test_creation(self, sample_laps):
        """Test creating a LapCollection"""
        collection = LapCollection(laps=sample_laps)
        assert len(collection) == 3
    
    def test_to_dataframe(self, sample_laps):
        """Test exporting to DataFrame"""
        collection = LapCollection(laps=sample_laps)
        df = collection.to_dataframe()
        
        assert len(df) == 3
        assert 'rider_name' in df.columns
        assert 'lap_time_ms' in df.columns
        assert 'top_speed' in df.columns
    
    def test_best_lap_overall(self, sample_laps):
        """Test finding best lap overall"""
        collection = LapCollection(laps=sample_laps)
        best = collection.best_lap()
        
        assert best is not None
        assert best.rider_name == 'Bagnaia'
        assert best.lap_number == 2
    
    def test_best_lap_by_rider(self, sample_laps):
        """Test finding best lap for specific rider"""
        collection = LapCollection(laps=sample_laps)
        best = collection.best_lap('Bagnaia')
        
        assert best.rider_name == 'Bagnaia'
        assert best.lap_number == 2
    
    def test_laps_by_rider(self, sample_laps):
        """Test getting all laps for a rider"""
        collection = LapCollection(laps=sample_laps)
        bagnaia_laps = collection.laps_by_rider('Bagnaia')
        
        assert len(bagnaia_laps) == 2
        assert all(lap.rider_name == 'Bagnaia' for lap in bagnaia_laps)
    
    def test_compare_riders(self, sample_laps):
        """Test comparing two riders"""
        collection = LapCollection(laps=sample_laps)
        comparison = collection.compare_riders('Bagnaia', 'Márquez', metric='lap_time')
        
        assert comparison['rider1'] == 'Bagnaia'
        assert comparison['rider2'] == 'Márquez'
        assert 'delta_ms' in comparison
        assert 'ahead' in comparison
    
    def test_consistency_metric(self, sample_laps):
        """Test calculating consistency metrics"""
        collection = LapCollection(laps=sample_laps)
        stats = collection.consistency_metric('Bagnaia')
        
        assert stats['rider'] == 'Bagnaia'
        assert stats['count'] == 2
        assert 'std_dev_ms' in stats
        assert 'coefficient_of_variation' in stats
    
    def test_to_dict(self, sample_laps):
        """Test exporting to dictionary"""
        collection = LapCollection(laps=sample_laps)
        data = collection.to_dict()
        
        assert isinstance(data, list)
        assert len(data) == 3
        assert all('rider_name' in lap for lap in data)
    
    def test_to_json(self, sample_laps):
        """Test exporting to JSON"""
        collection = LapCollection(laps=sample_laps)
        json_str = collection.to_json()
        
        assert isinstance(json_str, str)
        assert 'Bagnaia' in json_str


class TestLap:
    """Tests for individual Lap"""
    
    def test_lap_creation(self):
        """Test creating a lap"""
        lap = Lap(
            lap_number=1,
            rider_name='Bagnaia',
            rider_number=1,
            lap_time=timedelta(seconds=99, milliseconds=500),
            sector_times=[
                timedelta(seconds=32),
                timedelta(seconds=33),
                timedelta(seconds=34, milliseconds=500),
            ],
            top_speed=320.0,
        )
        
        assert lap.lap_number == 1
        assert lap.rider_name == 'Bagnaia'
        assert lap.top_speed == 320.0
    
    def test_lap_to_dict(self):
        """Test lap dictionary conversion"""
        lap = Lap(
            lap_number=1,
            rider_name='Bagnaia',
            rider_number=1,
            lap_time=timedelta(seconds=99, milliseconds=123),
            sector_times=[
                timedelta(seconds=32),
                timedelta(seconds=33),
                timedelta(seconds=34, milliseconds=123),
            ],
            top_speed=320.5,
        )
        
        data = lap.to_dict()
        assert data['rider_name'] == 'Bagnaia'
        assert 'lap_time_str' in data
        assert 'sector_times' in data


class TestRider:
    """Tests for Rider"""
    
    def test_rider_creation(self):
        """Test creating a rider"""
        team = Team(name='Ducati', country='Italy')
        rider = Rider(
            name='Pecco Bagnaia',
            number=1,
            nationality='ITA',
            team=team,
        )
        
        assert rider.name == 'Pecco Bagnaia'
        assert rider.number == 1
        assert rider.team.name == 'Ducati'
    
    def test_rider_from_api_response(self):
        """Test parsing rider from API response"""
        api_data = {
            'name': 'Pecco Bagnaia',
            'number': 1,
            'nationality': 'ITA',
            'team': {
                'name': 'Ducati',
                'country': 'Italy',
                'color': '#DC0000',
            },
        }
        
        rider = Rider.from_api_response(api_data)
        assert rider.name == 'Pecco Bagnaia'
        assert rider.team.name == 'Ducati'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
