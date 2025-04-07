from llm_experiments.slack_tools import SlackTools


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
    print(f"status: {status}, channel: {channel}, ts: {ts}")
    