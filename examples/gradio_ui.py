from smolagents import GradioUI, ToolCallingAgent, TransformersModel, WebSearchTool


agent = ToolCallingAgent(
    tools=[WebSearchTool()],
    model=TransformersModel(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    verbosity_level=1,
    # planning_interval=3,
    name="example_agent",
    description="This is an example agent.",
    step_callbacks=[],
    stream_outputs=True,
)

GradioUI(agent, file_upload_folder="./data").launch()
