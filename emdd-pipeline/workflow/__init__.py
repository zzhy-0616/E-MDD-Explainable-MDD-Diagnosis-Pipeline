"""EEGPT MDD pipeline workflow orchestration."""

from workflow.config import load_config
from workflow.runner import WorkflowRunner

__all__ = ["load_config", "WorkflowRunner"]
