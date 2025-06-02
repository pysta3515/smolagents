import argparse
from io import BytesIO
from time import sleep

import helium
import os
import PIL.Image
from PIL import ImageDraw
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from helium import go_to

from smolagents import ToolCallingAgent, WebSearchTool, tool
from smolagents.agents import ActionStep
from smolagents.cli import load_model
from smolagents.models import InferenceClientModel


github_request = """
I'm trying to find how hard I have to work to get a repo in github.com/trending.
Can you navigate to the profile for the top author of the top trending repo, and give me their total number of commits over the last year?
"""  # The agent is able to achieve this request only when powered by GPT-4o or Claude-3.5-sonnet.

search_request = """
Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence containing the word "1992" that mentions a construction accident.
"""

meteo_request = """
Please navigate to a weather website and tell me the current weather conditions and forecast for Paris, France.
Include the temperature, precipitation chances, and general weather conditions for today.
"""



def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a web browser automation script with a specified model.")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",  # Makes it optional
        default=meteo_request,
        help="The prompt to run with the agent",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="LiteLLMModel",
        help="The model type to use (e.g., OpenAIServerModel, LiteLLMModel, TransformersModel, InferenceClientModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-4o",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="The inference provider to use for the model",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        help="The API base to use for the model",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="The API key to use for the model",
    )
    return parser.parse_args()


def save_screenshot(memory_step: ActionStep, agent: ToolCallingAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = memory_step.step_number
    if driver is not None:
        for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = PIL.Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
    return



@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver.back()

@tool
def click(x: int, y: int) -> None:
    """
    Click on a position on the screen
    
    Args:
        x: num pixels from the left edge
        y: num pixels from the top edge
    """
    # Reset mouse position first
    from helium import click as helium_click
    import helium
    driver = helium.get_driver()
    scroll_x = driver.execute_script("return window.scrollX;")
    scroll_y = driver.execute_script("return window.scrollY;")
    loc = helium.Point(scroll_x + x, scroll_y + y)
    helium_click(loc)



@tool
def go_to(page: str) -> None:
    """
    Navigate to a webpage.
    
    Args:
        page: URL of the page to navigate to
    """
    from helium import go_to as helium_go_to
    helium_go_to(page)

@tool
def type_text(text: str) -> None:
    """
    Type text into the currently focused element.
    
    Args:
        text: The text to type
    """
    webdriver.ActionChains(driver).send_keys(text).perform()


@tool
def scroll_page_down() -> None:
    """
    Scroll down the page by one page length.
    
    Args:
        num_pixels: Number of pixels to scroll down
    """
    webdriver.ActionChains(driver).send_keys(Keys.PAGE_DOWN).perform()

@tool
def scroll_page_up() -> None:
    """
    Scroll up the page by one page length.
    
    Args:
        num_pixels: Number of pixels to scroll down
    """
    webdriver.ActionChains(driver).send_keys(Keys.PAGE_UP).perform()


@tool
def close_popups() -> None:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()


def initialize_driver():
    """Initialize the Selenium WebDriver."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")
    return helium.start_chrome(headless=False, options=chrome_options)


def initialize_agent(model):
    """Initialize the CodeAgent with the specified model."""
    return ToolCallingAgent(
        tools=[go_to, type_text, scroll_page_down, scroll_page_up, click, close_popups, search_item_ctrl_f],
        model=model,
        step_callbacks=[save_screenshot],
        max_steps=20,
        verbosity_level=2,
    )


helium_instructions = """
Use your web_search tool when you want to get Google search results.
Then you can use helium to access websites. Don't use helium for Google search, only for navigating websites!
Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
First, go to the relevant page using go_to:
Action:
{
    "name": "go_to",
    "arguments": {"page": "github.com/trending"}
}

You can directly click clickable elements by inputting a click position as Click(x, y) with x num pixels from the left edge and y num pixels from the top edge.
Action:
{
    "name": "click",
    "arguments": {"x": 352, "y": 348}
}

To type text into a field after click on, use type_text:
Action:
{
    "name": "type_text", 
    "arguments": {"text": "Hello world"}
}



If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.

To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
Action:
{
    "name": "scroll_down",
    "arguments": {"num_pixels": 1200}
}

When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Action:
{
    "name": "close_popups",
    "arguments": {}
}

Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
Action:
{
    "name": "final_answer",
    "arguments": "YOUR_ANSWER_HERE"
}

If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
To list elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually, or use your tool search_item_ctrl_f.
Of course, you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url.
But beware that the screenshot will only be taken at the end of the whole action, it won't see intermediate states.
Don't kill the browser.
When you have modals or cookie banners on screen, you should get rid of them before you can click anything else.
Always output a single action at a time, using ONLY the format shown below.

"""


def run_webagent(
    prompt: str,
    model_type: str,
    model_id: str,
    provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> None:
    # Load environment variables
    load_dotenv()
    os.makedirs("screenshots", exist_ok=True)

    # Initialize the model based on the provided arguments
    # model = load_model(model_type, model_id, provider=provider, api_base=api_base, api_key=api_key)
    model = InferenceClientModel(
        model_id="https://oa9u6uq0urded73f.us-east-1.aws.endpoints.huggingface.cloud",
        max_tokens=4096,
    )

    global driver
    driver = initialize_driver()
    agent = initialize_agent(model)

    # Run the agent with the provided prompt
    agent.run(prompt + helium_instructions)


def main() -> None:
    # Parse command line arguments
    args = parse_arguments()
    run_webagent(args.prompt, args.model_type, args.model_id, args.provider, args.api_base, args.api_key)


if __name__ == "__main__":
    main()
