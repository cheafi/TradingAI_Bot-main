# src/config.py
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class Config:
 # Account / sizing
 INITIAL_CAPITAL: float = 50_000.0
 KELLY_P: float = 0.60
 KELLY_B: float = 1.5
 KELLY_CAP: float = 0.01

 # Indicators
 EMA_PERIOD: int = 20
 ATR_PERIOD: int = 14
 KELTNER_MULT: float = 1.5
 RSI_FAST: int = 3

 # Stops / trailing
 K_INIT: float = 1.0
 P_BE: float = 0.6
 K_TRAIL: float = 0.8
 K_FAST: float = 0.6
 R1: float = 1.0
 R2: float = 2.0
 MAX_HOLD_BARS: int = 24

 # Data
 DEFAULT_TIMEFRAME: str = "15m"
 DEFAULT_LIMIT: int = 1500

 # Environment toggles
 TELEGRAM_ENABLED: bool = False

 # Trading settings
 mode: str = "demo"
 symbol: Optional[str] = None

 # AI / Azure OpenAI
 azure_tenant_id: Optional[str] = None
 azure_client_id: Optional[str] = None
 azure_client_secret: Optional[str] = None
 azure_openai_endpoint: Optional[str] = None
 azure_openai_deployment: Optional[str] = None
 azure_openai_api_version: Optional[str] = None

 def __post_init__(self):
  if self.symbol is None:
   self.symbol = "BTC/USDT"
  # Load Azure settings
  self.azure_tenant_id = os.getenv("AZURE_TENANT_ID")
  self.azure_client_id = os.getenv("AZURE_CLIENT_ID")
  self.azure_client_secret = os.getenv("AZURE_CLIENT_SECRET")
  self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
  self.azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
  self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

cfg = Config()