"""Wikipedia search and crawl tool definition."""

from pathlib import Path

from pydantic import BaseModel, Field

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Search or crawl Wikipedia articles using a retrieval service.

Commands:
- `search`: Search for relevant Wikipedia passages given a list of queries.
- `crawl`: Fetch full passages from specific Wikipedia URLs.

Environment variables RETRIEVAL_SERVICE_URL and CRAWL_SERVICE_URL must be set
to point to the LocalWiki retrieval service endpoints.
""".strip()


class SearchWikiArguments(BaseModel):
    command: str = Field(
        description="The command to run. Allowed options are: `search`, `crawl`.",
        json_schema_extra={"enum": ["search", "crawl"]},
    )
    query_list: list[str] = Field(
        default=None,
        description="Required for `search` command. A list of search queries to retrieve relevant Wikipedia passages.",
    )
    url_list: list[str] = Field(
        default=None,
        description="Required for `crawl` command. A list of Wikipedia URLs to fetch passages from.",
    )
    topk: int = Field(
        default=3,
        description="Optional for `search` command. Number of top results to return per query.",
    )


@register_tool("search")
class SearchWikiTool(AbstractTool):
    @property
    def name(self) -> str:
        return "search"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "search"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=SearchWikiArguments,
        )

    def get_install_command(self) -> str | None:
        return None
