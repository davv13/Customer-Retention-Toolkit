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

class CustomerMetrics(Base):
    """
    Represents the CustomerMetrics table in the database, holding customer churn metrics.

    Attributes:
        CustomerID (int): Primary key. Unique identifier for a customer.
        StateID (int): Foreign key to the State table. Represents the state ID associated with the customer.
        PlanID (int): Foreign key to the PlanDetails table. Represents the plan ID associated with the customer.
        DayUsageID (int): Foreign key to the DayUsage table. Represents the day usage ID associated with the customer.
        EveUsageID (int): Foreign key to the EveUsage table. Represents the evening usage ID associated with the customer.
        NightUsageID (int): Foreign key to the NightUsage table. Represents the night usage ID associated with the customer.
        IntlUsageID (int): Foreign key to the IntlUsage table. Represents the international usage ID associated with the customer.
        CustomerServiceCalls (int): Number of customer service calls made by the customer.
        ChurnStatus (str): The churn status of the customer.
    """

    __tablename__ = "CustomerMetrics"

    CustomerID = Column(Integer, primary_key=True)
    StateID = Column(Integer, ForeignKey('State.StateID'))
    PlanID = Column(Integer, ForeignKey('PlanDetails.PlanID'))
    DayUsageID = Column(Integer, ForeignKey('DayUsage.DayUsageID'))
    EveUsageID = Column(Integer, ForeignKey('EveUsage.EveUsageID'))
    NightUsageID = Column(Integer, ForeignKey('NightUsage.NightUsageID'))
    IntlUsageID = Column(Integer, ForeignKey('IntlUsage.IntlUsageID'))
    CustomerServiceCalls = Column(Integer)
    ChurnStatus = Column(String)

class State(Base):
    """
    Represents the State table in the database.

    Attributes:
        StateID (int): Primary key. Unique identifier for a state.
        StateName (str): The name of the state.
    """

    __tablename__ = "State"

    StateID = Column(Integer, primary_key=True, autoincrement=True)
    StateName = Column(String)

class PlanDetails(Base):
    """
    Represents the PlanDetails table in the database, holding information about different plans.

    Attributes:
        PlanID (int): Primary key. Unique identifier for a plan.
        AreaCode (int): The area code associated with the plan.
        InternationalPlan (str): Indicates if the plan includes international services.
        VoiceMailPlan (str): Indicates if the plan includes voicemail services.
        NumberVMailMessages (int): Number of voicemail messages included in the plan.
    """

    __tablename__ = "PlanDetails"

    PlanID = Column(Integer, primary_key=True, autoincrement=True)
    AreaCode = Column(Integer)
    InternationalPlan = Column(String)
    VoiceMailPlan = Column(String)
    NumberVMailMessages = Column(Integer)


class DayUsage(Base):
    """
    Represents the DayUsage table in the database, holding information about customers' day usage.

    Attributes:
        DayUsageID (int): Primary key. Unique identifier for a day usage record.
        TotalDayMinutes (float): Total minutes used by the customer during the day.
        TotalDayCalls (int): Total number of calls made by the customer during the day.
        TotalDayCharge (float): Total charge for the customer's day usage.
    """

    __tablename__ = "DayUsage"

    DayUsageID = Column(Integer, primary_key=True, autoincrement=True)
    TotalDayMinutes = Column(Float)
    TotalDayCalls = Column(Integer)
    TotalDayCharge = Column(Float)


class EveUsage(Base):
    """
    Represents the EveUsage table in the database, holding information about customers' evening usage.

    Attributes:
        EveUsageID (int): Primary key. Unique identifier for an evening usage record.
        TotalEveMinutes (float): Total minutes used by the customer during the evening.
        TotalEveCalls (int): Total number of calls made by the customer during the evening.
        TotalEveCharge (float): Total charge for the customer's evening usage.
    """

    __tablename__ = "EveUsage"

    EveUsageID = Column(Integer, primary_key=True, autoincrement=True)
    TotalEveMinutes = Column(Float)
    TotalEveCalls = Column(Integer)
    TotalEveCharge = Column(Float)


class NightUsage(Base):
    """
    Represents the NightUsage table in the database, holding information about customers' night usage.

    Attributes:
        NightUsageID (int): Primary key. Unique identifier for a night usage record.
        TotalNightMinutes (float): Total minutes used by the customer during the night.
        TotalNightCalls (int): Total number of calls made by the customer during the night.
        TotalNightCharge (float): Total charge for the customer's night usage.
    """

    __tablename__ = "NightUsage"

    NightUsageID = Column(Integer, primary_key=True, autoincrement=True)
    TotalNightMinutes = Column(Float)
    TotalNightCalls = Column(Integer)
    TotalNightCharge = Column(Float)


class IntlUsage(Base):
    """
    Represents the IntlUsage table in the database, holding information about customers' international usage.

    Attributes:
        IntlUsageID (int): Primary key. Unique identifier for an international usage record.
        TotalIntlMinutes (float): Total minutes used by the customer for international calls.
        TotalIntlCalls (int): Total number of international calls made by the customer.
        TotalIntlCharge (float): Total charge for the customer's international usage.
    """

    __tablename__ = "IntlUsage"

    IntlUsageID = Column(Integer, primary_key=True, autoincrement=True)
    TotalIntlMinutes = Column(Float)
    TotalIntlCalls = Column(Integer)
    TotalIntlCharge = Column(Float)


class PredictionResults(Base):
    """
    Represents the PredictionResults table in the database, holding the results of churn predictions.

    Attributes:
        PredictionID (int): Primary key. Unique identifier for a prediction record.
        CustomerID (int): Foreign key to the CustomerMetrics table. Identifies the customer for whom the prediction is made.
        PredictedLabel (str): The predicted label (e.g., 'Churn' or 'No Churn').
        ModelName (str): The name of the model used for the prediction.
        ChurnStatus (int): Foreign key to the CustomerMetrics table. Indicates the actual churn status of the customer.
    """

    __tablename__ = "PredictionResults"

    PredictionID = Column(Integer, primary_key=True)
    CustomerID = Column(Integer, ForeignKey('CustomerMetrics.CustomerID'))
    PredictedLabel = Column(String)
    ModelName = Column(String)
    ChurnStatus = Column(Integer, ForeignKey('CustomerMetrics.ChurnStatus'))

def create_database():
    """
    Creates all tables in the database based on the defined schema.

    This function initializes the database by creating all the tables defined in the schema if they do not already exist.
    """
    Base.metadata.create_all(engine)