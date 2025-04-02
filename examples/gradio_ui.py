from smolagents import CodeAgent, GradioUI, HfApiModel


def add_agent_image(memory_step, agent):
    # Actually loads an image from a url
    from io import BytesIO

    import requests
    from PIL import Image

    url = "https://upload.wikimedia.org/wikipedia/commons/3/3f/JPEG_example_flower.jpg"
    response = requests.get(url)
    memory_step.observations_images = [Image.open(BytesIO(response.content))]


agent = CodeAgent(
    tools=[],
    model=HfApiModel(provider="together"),
    verbosity_level=1,
    # planning_interval=2,
    name="example_agent",
    description="This is an example agent that has not tool but will always see an agent at the end of its step.",
    step_callbacks=[add_agent_image],
)

GradioUI(agent, file_upload_folder="./data").launch()
agent.run("Make me an image of a cow")
print("OBSERVATION IMAGES:", agent.memory.steps)
