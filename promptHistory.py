from typing import List

from pydantic import BaseModel
from prompt import Prompt


class PromptHistory(BaseModel):
    prompt_history: List[Prompt]
