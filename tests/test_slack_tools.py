from llm_experiments.slack_tools import SlackTools
from llm_experiments.tools import make_tools_by_name


def test_slack_tools():
    slack_tools = SlackTools()
    res = slack_tools.post_message("test", "hello from agent")
    assert res["ok"]

    res = slack_tools.update_message(res["channel"], res["ts"], "hello from agent updated")
    assert res["ok"]

    res = slack_tools.delete_message(res["channel"], res["ts"])
    assert res["ok"]

    tool_by_name = make_tools_by_name(slack_tools.tools)
    res = tool_by_name["post"].invoke({"channel": "test", "text": "hello from agent"})
    assert res["ok"]

    res = tool_by_name["delete"].invoke({"channel": res["channel"], "ts": res["ts"]})
    assert res["ok"]
