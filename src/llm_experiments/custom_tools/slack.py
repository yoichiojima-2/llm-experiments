import os

import slack_sdk
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool, BaseToolkit


class SlackToolkit(BaseToolkit):
    @property
    def client(self):
        return slack_sdk.WebClient(token=os.getenv("SLACK_USER_TOKEN"))

    def get_tools(self) -> list[BaseTool]:
        @tool
        def post_message(channel: str, text: str):
            """post a message to a Slack channel."""
            return self.client.chat_postMessage(channel=channel, text=text)

        @tool
        def delete_message(channel: str, ts: str):
            """delete a message from a Slack channel."""
            return self.client.chat_delete(channel=channel, ts=ts)

        @tool
        def post_ephemeral(channel: str, text: str, user: str):
            """post an ephemeral message to a Slack channel."""
            return self.client.chat_postEphemeral(channel=channel, text=text, user=user)

        @tool
        def update_message(channel: str, ts: str, text: str):
            """update a message in a Slack channel."""
            return self.client.chat_update(channel=channel, ts=ts, text=text)

        @tool
        def add_reaction(channel: str, emoji_name: str, ts: str):
            """add a reaction to a message in a Slack channel."""
            return self.client.reactions_add(channel=channel, name=emoji_name, timestamp=ts)

        @tool
        def remove_reaction(channel: str, emoji_name: str, ts: str):
            """remove a reaction from a message in a Slack channel."""
            return self.client.reactions_remove(channel=channel, name=emoji_name, timestamp=ts)

        @tool
        def upload_file(channels: str, file: str):
            """upload a file to a Slack channel."""
            return self.client.files_upload_v2(channels=channels, file=file)

        @tool
        def add_remote_file(channels: list[str], file: str):
            """add a remote file to a Slack channel."""
            return self.client.files_remote_add(channels=channels, file=file)

        @tool
        def list_conversations(limit: int = 100):
            """list all conversations in Slack."""
            return self.client.conversations_list(limit=limit)

        @tool
        def get_conversation_history(channel: str):
            """get the history of a conversation in Slack."""
            return self.client.conversations_history(channel=channel)

        @tool
        def start_direct_message(users: list[str]):
            """open a conversation in Slack."""
            return self.client.conversations_open(users=users)

        @tool
        def create_channel(name: str):
            """create a new channel in Slack."""
            return self.client.conversations_archive(name=name)

        @tool
        def get_conversation_info(channel: str):
            """get information about a conversation in Slack."""
            return self.client.conversations_info(channel=channel)

        @tool
        def get_members_of_conversation(channel: str):
            """get members of a conversation in Slack."""
            return self.client.conversations_members(channel=channel)

        @tool
        def join_conversation(channel: str):
            """join a conversation in Slack."""
            return self.client.conversations_join(channel=channel)

        @tool
        def leave_conversation(channel: str):
            """leave a conversation in Slack."""
            return self.client.conversations_leave(channel=channel)

        return [
            post_message,
            delete_message,
            post_ephemeral,
            update_message,
            add_reaction,
            remove_reaction,
            upload_file,
            add_remote_file,
            list_conversations,
            get_conversation_history,
            start_direct_message,
            create_channel,
            get_conversation_info,
            get_members_of_conversation,
            join_conversation,
            leave_conversation,
        ]
