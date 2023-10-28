
# from ..logger import CustomFormatter

import logging
import os

import logging
from .etl.logger import CustomFormatter

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


from sqlalchemy import create_engine,Column,Integer,String,Float, DATE, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

engine=create_engine('sqlite:///temp.db')

Base= declarative_base()

class Employee(Base):
    __tablename__ = "ChurnMetrics"

    customerID = Column(Integer, primary_key=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String)
    salary = Column(Float)

class Gender(Base):
    __tablename__ = "Gender"

    GenderID = Column(Integer, primary_key=True)
    Gender= Column(String)


class Contract(Base):
    __tablename__ = "Contract"

    ContractID = Column(Integer, primary_key=True)
    ContractType = Column(String)
    PareplessBilling = Column(String)

# Define the Order table
class PaymentMethod(Base):
    __tablename__ = "PaymentMethod"
    
    order_id = Column(Integer, primary_key=True)
    order_date = Column(DateTime)
    year = Column(Integer)
    quarter = Column(Integer)
    month = Column(String)

# Define the Sales table
class Sale(Base):
    __tablename__ = "sales"

    transaction_id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.order_id'))
    product_id = Column(Integer, ForeignKey('products.product_id'))
    customer_id = Column(Integer, ForeignKey('customers.customer_id'))
    employee_id = Column(Integer, ForeignKey('employees.employee_id'))
    total_sales = Column(Float)
    quantity = Column(Integer)
    discount = Column(Float)

    order = relationship("Order")
    product = relationship("Product")
    customer = relationship("Customer")
    employee = relationship("Employee")


Base.metadata.create_all(engine)