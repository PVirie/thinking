from fastapi import APIRouter, FastAPI, Depends, HTTPException, Header, Query, Body, status, Response, Form, Request
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Any
from pydantic import BaseModel
import os
import json
import aiohttp
import asyncio
from loguru import logger

dir_path = os.path.dirname(os.path.realpath(__file__))
logfile = os.path.join(dir_path, "..", "log", "log.log")
logger.add(logfile, rotation="500 MB")

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
    out_text = "Hello, "
    async with aiohttp.ClientSession() as session:
        async with session.get("http://metric-host:9000/api/v1/test") as r:
            if r.status == 200:
                out_text += await r.json()
    return out_text


@ app.on_event("startup")
async def startup_event():
    logger.info("Test is starting up.")


@ app.on_event("shutdown")
def shutdown_event():
    logger.info("Test is exiting.", "Wait a moment until completely exits.")


app.include_router(router)
app.mount("/", StaticFiles(directory="."))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
