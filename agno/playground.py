from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.playground import Playground, serve_playground_app
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

from dotenv import load_dotenv
from os import getenv

# Chargement des variables d'environnement
load_dotenv()

agent_storage: str = "tmp/agents.db"

web_agent = Agent(
    name="Agent Web",
    model=OpenAILike(
        id=getenv("NEBIUS_AI_MODEL"),
        api_key=getenv("NEBIUS_API_KEY"),
        base_url=getenv("NEBIUS_API_BASE"),
        tool_choice="none",
    ),
    tools=[DuckDuckGoTools()],
    instructions=["Toujours inclure les sources"],
    # Store the agent sessions in a sqlite database
    storage=SqliteAgentStorage(table_name="web_agent", db_file=agent_storage),
    # Adds the current date and time to the instructions
    add_datetime_to_instructions=True,
    # Adds the history of the conversation to the messages
    add_history_to_messages=True,
    # Number of history responses to add to the messages
    num_history_responses=5,
    # Adds markdown formatting to the messages
    markdown=True,
)

finance_agent = Agent(
    name="Agent Finance",
    model=OpenAILike(
        id=getenv("NEBIUS_AI_MODEL"),
        api_key=getenv("NEBIUS_API_KEY"),
        base_url=getenv("NEBIUS_API_BASE"),
        tool_choice="none",
    ),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Toujours utiliser des tableaux pour afficher les données"],
    storage=SqliteAgentStorage(table_name="finance_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

app = Playground(agents=[web_agent, finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)