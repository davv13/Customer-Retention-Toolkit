import logging
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Logging setup
from ..logger import CustomFormatter

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# Database setup
engine = create_engine('sqlite:///temp.db')
Base = declarative_base()

# Define the ChurnMetrics table (Fact Table)
class ChurnMetrics(Base):
    __tablename__ = "ChurnMetrics"

    CustomerID = Column(Integer, primary_key=True)
    StateID = Column(Integer, ForeignKey('State.StateID'))
    PlanID = Column(Integer, ForeignKey('PlanDetails.PlanID'))
    DayUsageID = Column(Integer, ForeignKey('DayUsage.DayUsageID'))
    EveUsageID = Column(Integer, ForeignKey('EveUsage.EveUsageID'))
    NightUsageID = Column(Integer, ForeignKey('NightUsage.NightUsageID'))
    IntlUsageID = Column(Integer, ForeignKey('IntlUsage.IntlUsageID'))
    CustomerServiceCalls = Column(Integer)
    ChurnStatus = Column(String)

# Define the State table
class State(Base):
    __tablename__ = "State"

    StateID = Column(Integer, primary_key=True)
    StateName = Column(String)

# Define the PlanDetails table
class PlanDetails(Base):
    __tablename__ = "PlanDetails"

    PlanID = Column(Integer, primary_key=True)
    AreaCode = Column(Integer)
    InternationalPlan = Column(String)
    VoiceMailPlan = Column(String)
    NumberVMailMessages = Column(Integer)

# Define the DayUsage table
class DayUsage(Base):
    __tablename__ = "DayUsage"

    DayUsageID = Column(Integer, primary_key=True)
    TotalDayMinutes = Column(Float)
    TotalDayCalls = Column(Integer)
    TotalDayCharge = Column(Float)

# Define the EveUsage table
class EveUsage(Base):
    __tablename__ = "EveUsage"

    EveUsageID = Column(Integer, primary_key=True)
    TotalEveMinutes = Column(Float)
    TotalEveCalls = Column(Integer)
    TotalEveCharge = Column(Float)

# Define the NightUsage table
class NightUsage(Base):
    __tablename__ = "NightUsage"

    NightUsageID = Column(Integer, primary_key=True)
    TotalNightMinutes = Column(Float)
    TotalNightCalls = Column(Integer)
    TotalNightCharge = Column(Float)

# Define the IntlUsage table
class IntlUsage(Base):
    __tablename__ = "IntlUsage"

    IntlUsageID = Column(Integer, primary_key=True)
    TotalIntlMinutes = Column(Float)
    TotalIntlCalls = Column(Integer)
    TotalIntlCharge = Column(Float)

# Create all tables
def create_database():
    # Create all tables
    Base.metadata.create_all(engine)
#Base.metadata.create_all(engine)
