#!/usr/bin/env python3
"""
Telegram bot utilities for adversarial spec development.

Usage:
    python3 telegram_bot.py setup              # Setup instructions and chat_id discovery
    python3 telegram_bot.py send <<< "message" # Send message from stdin
    python3 telegram_bot.py poll --timeout 60  # Poll for reply

Environment:
    TELEGRAM_BOT_TOKEN - Bot token from @BotFather
    TELEGRAM_CHAT_ID   - Your chat ID (get via setup command)

Exit codes:
    0 - Success
    1 - Error (API failure, timeout, etc.)
    2 - Missing configuration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

TELEGRAM_API: str = "https://api.telegram.org/bot{token}/{method}"
MAX_MESSAGE_LENGTH: int = 4096


def get_config() -> tuple[str, str]:
    """Get bot token and chat ID from environment.

    Returns:
        Tuple of (token, chat_id). Empty strings if not set.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    return token, chat_id


def api_call(
    token: str, method: str, params: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Make Telegram Bot API call.

    Args:
        token: Bot API token.
        method: API method name (e.g., sendMessage, getUpdates).
        params: Optional query parameters.

    Returns:
        Parsed JSON response from Telegram API.

    Raises:
        RuntimeError: On HTTP or network errors.
    """
    url = TELEGRAM_API.format(token=token, method=method)
    if params:
        url += "?" + urlencode(params)

    try:
        req = Request(url, headers={"User-Agent": "adversarial-spec/1.0"})
        with urlopen(req, timeout=30) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"Telegram API error {e.code}: {body}")
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")


def send_message(token: str, chat_id: str, text: str) -> bool:
    """Send a single message.

    Args:
        token: Bot API token.
        chat_id: Target chat identifier.
        text: Message text (supports Markdown).

    Returns:
        True on success, False on failure.
    """
    result = api_call(
        token,
        "sendMessage",
        {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
    )
    return result.get("ok", False)


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split long message into chunks, preferring paragraph boundaries.

    Args:
        text: The message text to split.
        max_length: Maximum length per chunk.

    Returns:
        List of message chunks.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Try to split at paragraph boundary
        split_at = remaining.rfind("\n\n", 0, max_length)
        if split_at == -1 or split_at < max_length // 2:
            # Try single newline
            split_at = remaining.rfind("\n", 0, max_length)
        if split_at == -1 or split_at < max_length // 2:
            # Fall back to space
            split_at = remaining.rfind(" ", 0, max_length)
        if split_at == -1:
            # Hard split
            split_at = max_length

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()

    return chunks


def send_long_message(token: str, chat_id: str, text: str) -> bool:
    """Send message, splitting if necessary.

    Args:
        token: Bot API token.
        chat_id: Target chat identifier.
        text: Message text (may exceed 4096 chars).

    Returns:
        True if all chunks sent successfully.
    """
    chunks = split_message(text)
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            header = f"[{i + 1}/{len(chunks)}]\n"
            chunk = header + chunk
        if not send_message(token, chat_id, chunk):
            return False
        if i < len(chunks) - 1:
            time.sleep(0.5)  # Rate limit
    return True


def get_last_update_id(token: str) -> int:
    """Get the ID of the most recent update.

    Args:
        token: Bot API token.

    Returns:
        The update_id of the most recent update, or 0 if none.
    """
    result = api_call(token, "getUpdates", {"limit": 1, "offset": -1})
    updates = result.get("result", [])
    if updates:
        return updates[-1]["update_id"]
    return 0


def poll_for_reply(
    token: str, chat_id: str, timeout: int = 60, after_update_id: int = 0
) -> Optional[str]:
    """Poll for a reply from the specified chat.

    Args:
        token: Bot API token.
        chat_id: Chat to poll for replies from.
        timeout: Maximum seconds to wait.
        after_update_id: Only consider updates after this ID.

    Returns:
        Message text if received within timeout, None otherwise.
    """
    start_time = time.time()
    offset = after_update_id + 1 if after_update_id else None

    while time.time() - start_time < timeout:
        remaining = int(timeout - (time.time() - start_time))
        if remaining <= 0:
            break

        params = {"timeout": min(remaining, 30)}
        if offset:
            params["offset"] = offset

        try:
            result = api_call(token, "getUpdates", params)
            updates = result.get("result", [])

            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message", {})
                msg_chat_id = str(message.get("chat", {}).get("id", ""))
                text = message.get("text", "")

                if msg_chat_id == chat_id and text:
                    # Clear processed updates
                    api_call(token, "getUpdates", {"offset": offset})
                    return text

        except RuntimeError:
            time.sleep(1)
            continue

    return None


def discover_chat_id(token: str) -> None:
    """Poll for messages and print chat IDs.

    Args:
        token: Bot API token.

    Runs until interrupted with Ctrl+C.
    """
    print("Waiting for messages... Send any message to your bot.")
    print("Press Ctrl+C to stop.\n")

    seen_chats = set()
    offset = None

    try:
        while True:
            params = {"timeout": 10}
            if offset:
                params["offset"] = offset

            result = api_call(token, "getUpdates", params)
            updates = result.get("result", [])

            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message", {})
                chat = message.get("chat", {})
                chat_id = chat.get("id")
                chat_type = chat.get("type", "unknown")
                username = chat.get("username", "")
                first_name = chat.get("first_name", "")

                if chat_id and chat_id not in seen_chats:
                    seen_chats.add(chat_id)
                    name = username or first_name or "Unknown"
                    print(f"Found chat: {name} ({chat_type})")
                    print(f"  TELEGRAM_CHAT_ID={chat_id}")
                    print()

    except KeyboardInterrupt:
        print("\nDone.")


def cmd_setup(args: argparse.Namespace) -> None:
    """Print setup instructions and discover chat ID.

    Args:
        args: Parsed command-line arguments.
    """
    token, chat_id = get_config()

    print("=" * 50)
    print("Telegram Bot Setup for Adversarial Spec")
    print("=" * 50)
    print()

    if not token:
        print("Step 1: Create a Telegram bot")
        print("  1. Open Telegram and message @BotFather")
        print("  2. Send /newbot and follow the prompts")
        print("  3. Copy the bot token")
        print("  4. Set: export TELEGRAM_BOT_TOKEN='your-token-here'")
        print()
        print("Then run this command again.")
        sys.exit(2)

    print("Step 1: Bot token [OK]")
    print()

    if not chat_id:
        print("Step 2: Get your chat ID")
        print("  1. Open Telegram and message your bot (any message)")
        print("  2. This script will detect your chat ID")
        print()
        discover_chat_id(token)
        print()
        print("Set: export TELEGRAM_CHAT_ID='your-chat-id'")
        sys.exit(0)

    print("Step 2: Chat ID [OK]")
    print()
    print("Configuration complete. Testing...")
    print()

    if send_message(token, chat_id, "Adversarial Spec bot connected."):
        print("Test message sent successfully.")
    else:
        print("Failed to send test message. Check your configuration.")
        sys.exit(1)


def cmd_send(args: argparse.Namespace) -> None:
    """Send message from stdin.

    Args:
        args: Parsed command-line arguments.
    """
    token, chat_id = get_config()
    if not token or not chat_id:
        print(
            "Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set",
            file=sys.stderr,
        )
        sys.exit(2)

    text = sys.stdin.read().strip()
    if not text:
        print("Error: No message provided via stdin", file=sys.stderr)
        sys.exit(1)

    if send_long_message(token, chat_id, text):
        print("Message sent.")
    else:
        print("Failed to send message.", file=sys.stderr)
        sys.exit(1)


def cmd_poll(args: argparse.Namespace) -> None:
    """Poll for reply.

    Args:
        args: Parsed command-line arguments (includes timeout).
    """
    token, chat_id = get_config()
    if not token or not chat_id:
        print(
            "Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set",
            file=sys.stderr,
        )
        sys.exit(2)

    last_update = get_last_update_id(token)
    print(f"Polling for reply (timeout: {args.timeout}s)...", file=sys.stderr)

    reply = poll_for_reply(token, chat_id, args.timeout, last_update)
    if reply:
        print(reply)
    else:
        print("No reply received.", file=sys.stderr)
        sys.exit(1)


def cmd_notify(args: argparse.Namespace) -> None:
    """Send round notification and poll for feedback.

    Args:
        args: Parsed command-line arguments (includes timeout).
    """
    token, chat_id = get_config()
    if not token or not chat_id:
        print(
            "Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set",
            file=sys.stderr,
        )
        sys.exit(2)

    # Read notification from stdin
    notification = sys.stdin.read().strip()
    if not notification:
        print("Error: No notification provided via stdin", file=sys.stderr)
        sys.exit(1)

    # Get last update ID before sending
    last_update = get_last_update_id(token)

    # Send notification
    notification += (
        f"\n\n_Reply within {args.timeout}s to add feedback, or wait to continue._"
    )
    if not send_long_message(token, chat_id, notification):
        print("Failed to send notification.", file=sys.stderr)
        sys.exit(1)

    # Poll for reply
    reply = poll_for_reply(token, chat_id, args.timeout, last_update)

    # Output as JSON
    result = {"notification_sent": True, "feedback": reply}
    print(json.dumps(result))


def main() -> None:
    """Entry point for the telegram_bot CLI."""
    parser = argparse.ArgumentParser(
        description="Telegram bot utilities for adversarial spec development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # setup
    setup_parser = subparsers.add_parser(
        "setup", help="Setup instructions and chat ID discovery"
    )
    setup_parser.set_defaults(func=cmd_setup)

    # send
    send_parser = subparsers.add_parser("send", help="Send message from stdin")
    send_parser.set_defaults(func=cmd_send)

    # poll
    poll_parser = subparsers.add_parser("poll", help="Poll for reply")
    poll_parser.add_argument(
        "--timeout", "-t", type=int, default=60, help="Timeout in seconds"
    )
    poll_parser.set_defaults(func=cmd_poll)

    # notify
    notify_parser = subparsers.add_parser(
        "notify", help="Send notification and poll for feedback"
    )
    notify_parser.add_argument(
        "--timeout", "-t", type=int, default=60, help="Timeout in seconds"
    )
    notify_parser.set_defaults(func=cmd_notify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
