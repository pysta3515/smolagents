from smolagents import GradioUI, OpenAIServerModel, ToolCallingAgent, WebSearchTool


agent = ToolCallingAgent(
    tools=[WebSearchTool()],
    model=OpenAIServerModel(model_id="gpt-4o"),
    verbosity_level=1,
    # planning_interval=3,
    name="example_agent",
    description="This is an example agent.",
    step_callbacks=[],
    stream_outputs=True,
)

GradioUI(agent, file_upload_folder="./data").launch()
