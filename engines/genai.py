from abc import ABC, abstractmethod

from .base import Base


class GenAI(Base, ABC):
    """Each GenAI model should inherit this class to use specific methods."""

    prompt: str = (
        'Act as a translation API. Output ONLY the translated text, no '
        'explanations, no additional commentary. Do not add any prefix or '
        'suffix to the translated content. Preserve all HTML tags (e.g., '
        '<b>, <em>, <a>, <span>) and translate inner text only. Do not '
        'translate content in <code>, <pre>, or placeholders like '
        '{{id_01234}}. Retain placeholders matching {{id_01234}} as is. '
        'URLs/addresses should be preserved as is. Do not omit any part of '
        'the content, even if it seems unimportant. Do not use Markdown '
        'formatting (such as **, *, #, _, or ~) in the translation.')

    line_protocol_instruction: str = (
        ' The input is a JSON object with "targetLanguage", "title", '
        '"description", and "segments" fields. Each segment has an "id" '
        'and "text". Output raw text lines in "ID | Text" format. Output '
        'exactly one line per segment using the format: '
        '"{id} | {translated_text}". Copy the exact "id" from the input '
        'segment to the output line. If the translated text contains a '
        'newline, replace it with the HTML tag "<br>" to ensure it stays '
        'on a single line. Use the pipe symbol " | " strictly to separate '
        'the ID and the text.')

    models: list[str]
    model: str | None
    samplings: list
    sampling: str
    temperature: float
    top_p: float
    top_k: int

    def get_prompt(self) -> str:
        return self.prompt + self.get_prompt_extension() \
            + self.line_protocol_instruction

    def get_prompt_extension(self) -> str:
        return ''

    @abstractmethod
    def get_models(self) -> list[str]:
        """Automatically get the models for the engine."""
