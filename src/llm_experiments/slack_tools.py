import os
import sys
from functools import wraps

import slack_sdk
from langchain_core.tools import StructuredTool, tool
from slack_sdk.errors import SlackApiError


def handle_error(func):
    @wraps(func)
    def wrapper(*a, **kw):
        try:
            return func(*a, **kw)
        except SlackApiError as e:
            print(f"error: {e.response['error']}", file=sys.stderr)
            return None

    return wrapper


class SlackTools:
    def __init__(self):
        self.client = slack_sdk.WebClient(token=os.getenv("SLACK_USER_TOKEN"))

    def get_tools(self):
        return [self.post_message_tool()]

    def post_message_tool(self) -> StructuredTool:
        @tool
        @handle_error
        def post_message(channel: str, text: str) -> tuple[bool, str, str]:
            """Post a message to a Slack channel."""
            res = self.client.chat_postMessage(channel=channel, text=text)
            print(f"message sent to {channel}: {text}")
            return res["ok"], res["channel"], res["ts"]

        return post_message

    @handle_error
    def post_ephemeral(self, channel: str, text: str, user: str) -> tuple[bool, str, str]:
        """Post an ephemeral message to a Slack channel."""
        res = self.client.chat_postEphemeral(channel=channel, text=text, user=user)
        print(f"ephemeral message sent to {channel}: {text}")
        return res["ok"], res["channel"], res["ts"]

    @handle_error
    def update_message(self, channel: str, ts: str, text: str) -> tuple[bool, str, str]:
        """Update a message in a Slack channel."""
        res = self.client.chat_update(channel=channel, ts=ts, text=text)
        print(f"message updated in {channel}: {text}")
        return res["ok"], res["channel"], res["ts"]

    @handle_error
    def delete_message(self, channel: str, ts: str) -> tuple[bool, str, str]:
        """Delete a message in a Slack channel."""
        res = self.client.chat_delete(channel=channel, ts=ts)
        print(f"message deleted in {channel}: {ts}")
        return res["ok"], res["channel"], res["ts"]

    @handle_error
    def add_reaction(self, channel: str, emoji_name: str, ts: str) -> tuple[bool, str, str]:
        """Add a reaction to a message in a Slack channel."""
        res = self.client.reactions_add(channel=channel, name=emoji_name, timestamp=ts)
        print(f"reaction added to {channel}: {emoji_name} at {ts}")
        return res["ok"], res["channel"], res["ts"]

    @handle_error
    def remove_reaction(self, channel, emoji_name: str, ts: str) -> tuple[bool, str, str]:
        """Remove a reaction from a message in a Slack channel."""
        res = self.client.reactions_remove(channel=channel, name=emoji_name, timestamp=ts)
        print(f"reaction removed from {channel}: {emoji_name} at {ts}")
        return res["ok"], res["channel"], res["ts"]

    @handle_error
    def upload_file(self, channels: str, file: str) -> tuple[bool, str, str]:
        """Upload a file to a Slack channel."""
        res = self.client.files_upload_v2(channels=channels, file=file)
        print(f"file uploaded to {channels}: {file}")
        return res["ok"], res["file"]["id"], res["file"]["name"]

    @handle_error
    def add_remote_file(self, channels: list[str], file: str) -> tuple[bool, str, str]:
        """Add a remote file to a Slack channel."""
        res = self.client.files_remote_add(channels=channels, file=file)
        print(f"remote file added to {channels}: {file}")
        return res["ok"], res["file"]["id"], res["file"]["name"]
