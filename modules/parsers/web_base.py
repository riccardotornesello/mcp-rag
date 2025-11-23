import logging
from abc import abstractmethod
from typing import Any

from bs4 import BeautifulSoup

# Assuming these are available in your environment
from datapizza.core.models import PipelineComponent
from datapizza.type import Node
from playwright.async_api import async_playwright

# Import Playwright
from playwright.sync_api import sync_playwright

# --- PARSER BASE CLASS ---


class Parser(PipelineComponent):
    """
    A parser is a pipeline component that converts a document into a structured hierarchical Node representation.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def parse(self, text: str, metadata: dict | None = None) -> Node:
        pass

    async def a_parse(self, text: str, metadata: dict | None = None) -> Node:
        raise NotImplementedError

    def _run(self, text: str, metadata: dict | None = None) -> Node:
        return self.parse(text, metadata)

    async def _a_run(self, text: str, metadata: dict | None = None) -> Node:
        return await self.a_parse(text, metadata)


# --- DYNAMIC WEB LOADER (PLAYWRIGHT) ---

logger = logging.getLogger(__name__)


class WebBaseLoader(Parser):
    """
    Loads a web page from a URL using a headless browser (Playwright).
    This supports Client-Side Rendering (React, Vue, Angular).
    """

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30000,
        bs_kwargs: dict[str, Any] | None = None,
    ):
        """
        Args:
            headless: Whether to run the browser in headless mode (no UI).
            timeout: Maximum time to wait for page load in milliseconds.
            bs_kwargs: Arguments passed to BeautifulSoup.
        """
        super().__init__()
        self.headless = headless
        self.timeout = timeout
        self.bs_kwargs = bs_kwargs or {}

    def _build_metadata(self, soup: BeautifulSoup, url: str) -> dict:
        """Extracts basic metadata from HTML."""
        metadata = {"source": url}
        if title := soup.find("title"):
            metadata["title"] = title.get_text()
        if description := soup.find("meta", attrs={"name": "description"}):
            metadata["description"] = description.get(
                "content", "No description found."
            )
        if html_tag := soup.find("html"):
            metadata["language"] = html_tag.get("lang", "No language found.")
        return metadata

    def _process_html(
        self, html_content: str, url: str, extra_metadata: dict | None
    ) -> Node:
        """Shared logic to clean HTML and create a Node."""
        soup = BeautifulSoup(html_content, "html.parser", **self.bs_kwargs)

        # Remove script and style elements to clean up text
        for script in soup(["script", "style"]):
            script.extract()

        # Clean text extraction
        text_content = soup.get_text(separator="\n\n", strip=True)

        # Metadata construction
        final_metadata = self._build_metadata(soup, url)
        if extra_metadata:
            final_metadata.update(extra_metadata)

        return Node(text=text_content, metadata=final_metadata)

    def parse(self, text: str, metadata: dict | None = None) -> Node:
        """
        Synchronously loads the URL using Playwright, waits for network idle, and extracts text.
        """
        url = text

        try:
            with sync_playwright() as p:
                # Launch browser
                browser = p.chromium.launch(headless=self.headless)
                page = browser.new_page()

                # Navigate and wait for the network to be idle (useful for SPAs)
                page.goto(url, timeout=self.timeout)
                # 'networkidle' is crucial for React: it waits until there are no network connections for at least 500 ms
                page.wait_for_load_state("networkidle")

                html_content = page.content()
                browser.close()

                return self._process_html(html_content, url, metadata)

        except Exception as e:
            logger.error(f"Error fetching {url} with Playwright: {e}")
            raise e

    async def a_parse(self, text: str, metadata: dict | None = None) -> Node:
        """
        Asynchronously loads the URL using Playwright, waits for network idle, and extracts text.
        """
        url = text

        try:
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=self.headless)
                page = await browser.new_page()

                # Navigate and wait for the network to be idle
                await page.goto(url, timeout=self.timeout)
                await page.wait_for_load_state("networkidle")

                html_content = await page.content()
                await browser.close()

                return self._process_html(html_content, url, metadata)

        except Exception as e:
            logger.error(f"Error asynchronously fetching {url} with Playwright: {e}")
            raise e
