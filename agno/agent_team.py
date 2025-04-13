from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.playground import Playground, serve_playground_app
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

from dotenv import load_dotenv
from os import getenv

# Chargement des variables d'environnement
load_dotenv()

web_agent = Agent(
    name="Agent Web",
    role="Rechercher des informations sur le web",
    model=OpenAILike(
        id=getenv("NEBIUS_AI_MODEL"),
        api_key=getenv("NEBIUS_API_KEY"),
        base_url=getenv("NEBIUS_API_BASE"),
        tool_choice="none",
    ),
    tools=[DuckDuckGoTools()],
    instructions=["Toujours inclure les sources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Agent Finance",
    role="Obtenir des données financières",
    model=OpenAILike(
        id=getenv("NEBIUS_AI_MODEL"),
        api_key=getenv("NEBIUS_API_KEY"),
        base_url=getenv("NEBIUS_API_BASE"),
        tool_choice="none",
    ),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Toujours utiliser des tableaux pour afficher les données"],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=OpenAILike(
        id=getenv("NEBIUS_AI_MODEL"),
        api_key=getenv("NEBIUS_API_KEY"),
        base_url=getenv("NEBIUS_API_BASE"),
        tool_choice="none",
    ),
    instructions=["Toujours inclure les sources", "Toujours utiliser des tableaux pour afficher les données"],
    show_tool_calls=True,
    markdown=True,
)

app = Playground(agents=[agent_team]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)

#agent_team.print_response("Quelle est la perspective du marché et la performance financière des entreprises de semi-conducteurs pour l'IA?", stream=True)
