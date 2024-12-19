import base64
import mimetypes
from pathlib import Path
from typing import TypedDict, Literal


class ImageContent(TypedDict):
    """Class for image content in messages"""
    type: Literal["image_url"]
    image_url: dict[str, str]


class TextContent(TypedDict):
    """Class for text content in messages"""
    type: Literal["text"]
    text: str


MessageContent = str | list[TextContent | ImageContent] | None


class BaseChatMessage(TypedDict):
    """Base class for chat messages, only role is always required"""
    role: Literal["user", "system", "assistant", "tool"]


class ChatMessage(BaseChatMessage, total=False):
    """Class for chat messages, all fields here are optional"""
    content: MessageContent
    tool_call_id: str | None
    name: str | None


def create_image_content(url: str,
                         detail: Literal["low", "high", "auto"] = "auto") -> ImageContent:
    """
    Helper function to create image content

    Args:
        url: Either a web URL or base64 encoded image (should start with 'data:image/jpeg;base64,')
        detail: Level of detail for image processing ("low", "high", or "auto")
    """
    return {
        "type": "image_url",
        "image_url": {
            "url": url,
            "detail": detail,
        }
    }


def create_local_image_content(
        file_path: str | Path,
        detail: Literal["low", "high", "auto"] = "auto"
) -> ImageContent:
    """
    Helper function to create image content from a local file

    Args:
        file_path: Path to the local image file (can be string or Path object)
        detail: Level of detail for image processing ("low", "high", or "auto")

    Returns:
        ImageContent object with base64-encoded image

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the file type is not supported
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    # Determine the MIME type of the image
    mime_type = mimetypes.guess_type(str(path))[0]
    if not mime_type or not mime_type.startswith('image/'):
        raise ValueError(f"Unsupported file type: {mime_type}")

    # Read and encode the image
    base64_image = base64.b64encode(path.read_bytes()).decode('utf-8')

    # Create the data URL with appropriate MIME type
    data_url = f"data:{mime_type};base64,{base64_image}"

    return {
        "type": "image_url",
        "image_url": {
            "url": data_url,
            "detail": detail,
        }
    }


def user(
        content: str,
        images: list[ImageContent] | ImageContent | None = None
) -> ChatMessage:
    """
    Fill a user message. Supports both simple text and images.

    Args:
        content: Text message content
        images: Optional image(s) to include with the message. Can be a single image or list of
            images.

    Returns:
        ChatMessage with the specified content
    """
    if images is None:
        return {"role": "user", "content": content}

    message_content: list[TextContent | ImageContent] = [{"type": "text", "text": content}]

    if isinstance(images, list):
        message_content.extend(images)
    else:
        message_content.append(images)

    return {"role": "user", "content": message_content}


def assistant(content: str) -> ChatMessage:
    """Fill an assistant message"""
    return {"role": "assistant", "content": content}


def system(content: str) -> ChatMessage:
    """Fill a system message"""
    return {"role": "system", "content": content}
