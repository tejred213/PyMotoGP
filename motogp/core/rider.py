"""Rider module - Rider and team information"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class Team:
    """
    Represents a MotoGP team.
    
    Attributes:
        name: Team name
        country: Home country
        color: Primary team color (hex)
        short_name: Team abbreviation
    """
    name: str
    country: Optional[str] = None
    color: Optional[str] = None
    short_name: Optional[str] = None
    
    def __repr__(self) -> str:
        return f"Team('{self.name}')"


@dataclass
class Rider:
    """
    Represents a MotoGP rider.
    
    Attributes:
        name: Full rider name
        number: Race number
        nationality: Country code
        team: Team object
        birth_date: Date of birth (optional)
        biography: Short bio (optional)
    """
    name: str
    number: int
    nationality: str
    team: Optional[Team] = None
    birth_date: Optional[str] = None
    biography: Optional[str] = None
    
    def __repr__(self) -> str:
        return f"Rider({self.number}, '{self.name}')"
    
    @classmethod
    def from_api_response(cls, data: dict) -> 'Rider':
        """Parse rider from API response"""
        team = None
        if 'team' in data and data['team']:
            team_data = data['team']
            team = Team(
                name=team_data.get('name', ''),
                country=team_data.get('country'),
                color=team_data.get('color'),
                short_name=team_data.get('short_name'),
            )
        
        return cls(
            name=data.get('name', ''),
            number=int(data.get('number', 0)),
            nationality=data.get('nationality', ''),
            team=team,
            birth_date=data.get('birth_date'),
            biography=data.get('biography'),
        )
