import uvicorn
import os
from customer_retention_toolkit.api import app

if __name__=="__main__":
    uvicorn.run(app,port=5000)