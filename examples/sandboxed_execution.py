from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel


model = HfApiModel()

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)
output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Local executor result:", output)

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, use_docker_executor=True)
output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Docker executor result:", output)

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, use_e2b_executor=True)
output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("E2B executor result:", output)
