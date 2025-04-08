from llm_experiments.slack_tools import SlackTools
from llm_experiments.tools import make_tools_by_name


def test_slack_tools():
    slack_tools = SlackTools()
    status, channel, ts = slack_tools.post_message("test", "hello from agent")
    assert status

    print(f"status: {status}, channel: {channel}, ts: {ts}")
    status, channel, ts = slack_tools.update_message(channel, ts, "hello from agent updated")
    assert status

    print(f"status: {status}, channel: {channel}, ts: {ts}")
    status, channel, ts = slack_tools.delete_message(channel, ts)
    assert status

    tool_by_name = make_tools_by_name(slack_tools.tools)
    status, channel, ts = tool_by_name["post"].invoke({"channel": "test", "text": "hello from agent"})
    assert status

    status, _, _ = tool_by_name["delete"].invoke({"channel": channel, "ts": ts})
    assert status
