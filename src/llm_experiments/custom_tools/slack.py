import os
import sys
from functools import wraps

import slack_sdk
from langchain_core.tools import tool
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

    @property
    def tools(self):
        @tool
        def post_message(channel: str, text: str):
            """post a message to a Slack channel."""
            return self.post_message(channel, text)

        @tool
        def delete_message(channel: str, ts: str):
            """delete a message from a Slack channel."""
            return self.delete_message(channel, ts)

        @tool
        def update_message(channel: str, ts: str, text: str):
            """update a message in a Slack channel."""
            return self.update_message(channel, ts, text)

        @tool
        def add_reaction(channel: str, emoji_name: str, ts: str):
            """add a reaction to a message in a Slack channel."""
            return self.add_reaction(channel, emoji_name, ts)

        @tool
        def remove_reaction(channel: str, emoji_name: str, ts: str):
            """remove a reaction from a message in a Slack channel."""
            return self.remove_reaction(channel, emoji_name, ts)

        @tool
        def upload_file(channels: str, file: str):
            """upload a file to a Slack channel."""
            return self.upload_file(channels, file)

        @tool
        def add_remote_file(channels: list[str], file: str):
            """add a remote file to a Slack channel."""
            return self.add_remote_file(channels, file)

        @tool
        def list_conversations():
            """list all conversations in Slack."""
            return self.list_conversations()

        return [
            post_message,
            delete_message,
            update_message,
            add_reaction,
            remove_reaction,
            upload_file,
            add_remote_file,
            list_conversations,
        ]

    @handle_error
    def post_message(self, channel: str, text: str):
        return self.client.chat_postMessage(channel=channel, text=text)

    @handle_error
    def post_ephemeral(self, channel: str, text: str, user: str):
        return self.client.chat_postEphemeral(channel=channel, text=text, user=user)

    @handle_error
    def update_message(self, channel: str, ts: str, text: str):
        return self.client.chat_update(channel=channel, ts=ts, text=text)

    @handle_error
    def delete_message(self, channel: str, ts: str):
        return self.client.chat_delete(channel=channel, ts=ts)

    @handle_error
    def add_reaction(self, channel: str, emoji_name: str, ts: str):
        return self.client.reactions_add(channel=channel, name=emoji_name, timestamp=ts)

    @handle_error
    def remove_reaction(self, channel, emoji_name: str, ts: str):
        return self.client.reactions_remove(channel=channel, name=emoji_name, timestamp=ts)

    @handle_error
    def upload_file(self, channels: str, file: str):
        return self.client.files_upload_v2(channels=channels, file=file)

    @handle_error
    def add_remote_file(self, channels: list[str], file: str):
        return self.client.files_remote_add(channels=channels, file=file)

    @handle_error
    def list_conversations(self):
        return self.client.conversations_list()
