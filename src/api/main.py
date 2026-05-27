from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from mlflow.tracking import MlflowClient
import torch
import joblib
import mlflow
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

