from fastapi import FastAPI
from src.config import settings
from src.routers.health import router as health_router
from src.routers.simulation import router as simulation_router
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

swagger_ui_params = {
    "operationsSorter": "alpha", 
    "tagsSorter": "alpha", 
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(
    title=settings.app_name,    
    description=settings.app_description,
    swagger_ui_parameters=swagger_ui_params,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(simulation_router)