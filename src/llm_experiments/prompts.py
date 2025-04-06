import textwrap

from langchain_core.prompts import PromptTemplate


def common_task_instruction() -> str:
    return textwrap.dedent(
        """
        You have access to the following tools:
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """
    )


def create_complete_prompt_template(query) -> PromptTemplate:
    return PromptTemplate.from_template(
        textwrap.dedent(
            f"""
            {query}
            {common_task_instruction()}
            """
        )
    )


def multipurpose() -> PromptTemplate:
    prompt = "Answer the following questions as best you can"
    return create_complete_prompt_template(prompt)
