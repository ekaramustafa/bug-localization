from pydantic import BaseModel
from typing import List


class OpenAILocalizerResponse(BaseModel):
    candidate_files: list[str]


class DirectorySelectionResponse(BaseModel):
    directories_to_explore: List[str] = []
    selected_files: List[str] = []


