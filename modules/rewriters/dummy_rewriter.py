from datapizza.core.modules.rewriter import Rewriter
from datapizza.memory.memory import Memory


class DummyRewriter(Rewriter):
    def rewrite(self, user_prompt: str, memory: Memory | None = None) -> str:
        return user_prompt
