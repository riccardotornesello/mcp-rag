from typing import Any

from datapizza.core.models import PipelineComponent


class TextLoader(PipelineComponent):
    def _run(self, path: str) -> Any:
        with open(path, encoding="utf-8") as f:
            return f.read()
