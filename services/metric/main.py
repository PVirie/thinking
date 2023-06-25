from fastapi import APIRouter, FastAPI, Depends, HTTPException, Header, Query, Body, status, Response, Form, Request
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Any
from pydantic import BaseModel
import os
import json
from loguru import logger

dir_path = os.path.dirname(os.path.realpath(__file__))
logfile = os.path.join(dir_path, "log", "log.log")

dir_path = os.path.dirname(os.path.realpath(__file__))

app = FastAPI(
    title="Think app",
    description="Think app",
    version="0.1.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)
router = APIRouter(prefix="/api/v1")


@ router.get("/test")
async def test():
    return "world"


@ app.on_event("startup")
async def startup_event():
    logger.add(logfile, rotation="500 MB")
    logger.info("Service is starting up.")


@ app.on_event("shutdown")
def shutdown_event():
    logger.info("Service is exiting.", "Wait a moment until completely exits.")


app.include_router(router)
app.mount("/", StaticFiles(directory="."))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9000, workers=1)
