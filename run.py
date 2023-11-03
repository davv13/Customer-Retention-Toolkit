import uvicorn
import os
from etl.api import app

if __name__=="__main__":
    uvicorn.run(app)