# llm-experiments
random experiments with llm

# usage
you need to run `ollama serve` to use `deepseek` and `llama`.

place a `.env` file at the root of the repository as shown in the example below:

```bash
# langsmith
LANGSMITH_TRACING=***
LANGSMITH_ENDPOINT=***
LANGSMITH_API_KEY=***
LANGSMITH_PROJECT=***
# openapi
OPENAI_API_KEY=***
# spotify
SPOTIPY_CLIENT_ID=***
SPOTIPY_CLIENT_SECRET=***
SPOTIPY_REDIRECT_URI=***
# serper
SERPER_API_KEY=***
# tavily
TAVILY_API_KEY=***
# google for gemini
GOOGLE_API_KEY=***
# slack
SLACK_USER_TOKEN=***
```

then run:
```bash
make install && source .env && source .venv/bin/activate
```

to start working with the swe team, run:
```bash
python -m llm_experiments --agent swe
```

if you just want a simple agent, run:
```bash
python -m llm_experiments
```

additionally, you can work with a more specialized agent by running:
```bash
python -m llm_experiments --agent slack
```

to see all available agent options, run:
```bash
python -m llm_experiments --help
```