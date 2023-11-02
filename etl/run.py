import uvicorn
import os
from api.api import app

if __name__=="__main__":
    uvicorn.run(app)