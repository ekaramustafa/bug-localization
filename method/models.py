from pydantic import BaseModel


class OpenAILocalizerResponse(BaseModel):
    candidate_files: list[str]
