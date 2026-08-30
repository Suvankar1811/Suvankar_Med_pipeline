from typing import Optional

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    """State carried through each phase of the data pipeline."""

    run_id: str = Field(default="", description="Unique identifier for the pipeline run")
    status: str = Field(default="initialized", description="Current status of the pipeline run")

    uploaded_files: list[str] = Field(
        default_factory=list,
        description="Paths to uploaded CSV files",
    )
    business_intent: str = Field(
        default="",
        description="The user's question in plain English",
    )

    profile_path: str = Field(
        default="",
        description="Path to the data profile JSON",
    )

    sttm_bronze_path: str = Field(description="Path to Bronze layer rules")
    sttm_silver_path: str = Field(description="Path to Silver layer rules")
    sttm_gold_path: str = Field(description="Path to Gold layer rules")
    hitl_approved: bool = Field(
        default=False,
        description="Whether a human approved the rules",
    )

    bronze_output_paths: list[str] = Field(
        description="Bronze Parquet file paths",
    )
    silver_output_paths: list[str] = Field(
        description="Silver Parquet file paths",
    )
    gold_output_paths: list[str] = Field(
        description="Gold Parquet file paths",
    )

    report_path: str = Field(
        default="",
        description="Path to the final HTML report",
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message captured during the pipeline run",
    )
