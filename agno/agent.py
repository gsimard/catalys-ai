from dotenv import load_dotenv
from os import getenv
from agno.agent import Agent, RunResponse
from agno.models.openai.like import OpenAILike

# Chargement des variables d'environnement
load_dotenv()

agent = Agent(
    model=OpenAILike(
        id=getenv("NEBIUS_AI_MODEL"),
        api_key=getenv("NEBIUS_API_KEY"),
        base_url=getenv("NEBIUS_API_BASE"),
    )
)

# Print the response in the terminal
agent.print_response("Raconte une histoire d'horreur en 2 phrases.")