
from pydantic import BaseModel, Field, ConfigDict


class MobileFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    battery_power: float = Field(ge=0)
    blue: int = Field(ge=0, le=1)
    clock_speed: float = Field(ge=0)
    dual_sim: int = Field(ge=0, le=1)
    fc: float = Field(ge=0)
    four_g: int = Field(ge=0, le=1)
    int_memory: float = Field(ge=0)
    m_dep: float = Field(ge=0)
    mobile_wt: float = Field(ge=0)
    n_cores: float = Field(ge=0)
    pc: float = Field(ge=0)
    px_height: float = Field(ge=0)
    px_width: float = Field(ge=0)
    ram: float = Field(ge=0)
    sc_h: float = Field(ge=0)
    sc_w: float = Field(ge=0)
    talk_time: float = Field(ge=0)
    three_g: int = Field(ge=0, le=1)
    touch_screen: int = Field(ge=0, le=1)
    wifi: int = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    predicted_price_range: int
    model_version: str


class BatchPredictionRequest(BaseModel):
    items: list[MobileFeatures] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    total_predictions: int
    predictions: list[PredictionResponse]
