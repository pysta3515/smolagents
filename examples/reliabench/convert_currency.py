import argparse
import random

from smolagents import ActionStep, CodeAgent, MultiStepAgent, OpenAIServerModel, Tool, ToolCallingAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Run currency conversion tests with different agent types")
    parser.add_argument("--n-steps", type=int, default=100, help="Number of currency conversions to perform")
    parser.add_argument(
        "--step-by-step", action="store_true", default=True, help="Whether to use step-by-step guidance"
    )
    parser.add_argument(
        "--agent-type", type=str, choices=["code", "json"], default="code", help="Type of agent to use (code or json)"
    )
    return parser.parse_args()


class StepByStepHandler:
    def __init__(self, step_by_step: bool, n_iterations: int, currencies_list: list[str]):
        self.step_by_step = step_by_step
        self.current_step = 1
        self.log_incorrect_conversions: list = []
        self.n_iterations = n_iterations
        self.currencies_list = currencies_list

    def callback(self, memory_step: ActionStep):
        if self.current_step < self.n_iterations:
            guidance = f"\n\nNEXT CONVERSION: Convert from {self.currencies_list[self.current_step]} to {self.currencies_list[self.current_step + 1]} => Do this now, don't do anything else!"
            self.current_step += 1
        else:
            guidance = "You have finished all the conversions. You can return the result now."
        if memory_step.observations:
            memory_step.observations += guidance
        else:
            memory_step.observations = guidance


class CurrencyConverter(Tool):
    name = "currency_converter"
    description = "Convert an amount of money from one currency to another"
    inputs = {
        "amount": {"type": "number", "description": "The amount of money to convert"},
        "from_currency": {"type": "string", "description": "The currency to convert from"},
        "to_currency": {"type": "string", "description": "The currency to convert to"},
    }
    output_type = "number"

    def __init__(self, currency_values: dict, tax: float = 1.02):
        super().__init__()
        self.conversions_done: list = []
        self.currency_values = currency_values
        self.tax = tax

    def forward(self, amount: float, from_currency: str, to_currency: str) -> float:
        self.conversions_done.append((amount, from_currency, to_currency))
        calculated_amount = (
            amount * self.currency_values[to_currency] / self.currency_values[from_currency] * (1 + self.tax)
        )
        return calculated_amount


def test_agent_conversions(
    agent: MultiStepAgent,
    n_conversions: int,
    currencies_list: list[str],
    currency_values: dict,
    step_by_step: bool = False,
    silent: bool = False,
) -> bool:
    original_value = random.uniform(100.0, 1000.0)

    agent_specific_instructions = (
        ("Return the number as a string, with only the number, no other text, like this: '1329953422.2198896'.")
        if isinstance(agent, ToolCallingAgent)
        else ""
    )

    if step_by_step:
        task = f"""You have been given an amount of money of {original_value} USD.
You will have to successively have it pass through {n_conversions} countries as an items travels, each time converting money and applying tax.
You have got access to the tool currency_converter, which will convert the amount of money and apply tax to handle the transfer from one currency to another. Use this to convert the amount of money from one currency to another.
This tool will also give you updated instructions in the logs on what conversion to do next. So whenever you don't know what to do next, just look at the logs from the previous tool use!
{agent_specific_instructions}

Here is the first conversion you will have to do:
Convert from {currencies_list[0]} to {currencies_list[1]}
"""
    else:
        task = f"""You have been given an amount of money of {original_value} USD.
You will have to successively have it pass through {n_conversions} countries as an items travels, each time converting money and applying tax.
You have got access to the tool currency_converter, which will convert the amount of money and apply tax to handle the transfer from one currency to another. Use this to convert the amount of money from one currency to another.
{agent_specific_instructions}

Here is the list of conversions you have to do:
"""
        for i in range(n_conversions):
            task += f"Convert from {currencies_list[i]} to {currencies_list[i + 1]}\n"

    test_currency_converter = CurrencyConverter(currency_values)
    expected_value = original_value
    for i in range(n_conversions):
        if not silent:
            print(f"Amount: {expected_value}, converting from {currencies_list[i]} to {currencies_list[i + 1]}")
        expected_value = test_currency_converter(expected_value, currencies_list[i], currencies_list[i + 1])
    task += "Now go on!"

    if not silent:
        print("Starting with value:", original_value)

    output = agent.run(task)
    try:
        final_value = float(output)  # type: ignore
    except ValueError:
        print("Output is not a number:", output)
        return False
    if not silent:
        print("Final value:", final_value)
        print("Expected value:", expected_value)
    if step_by_step:
        if not silent:
            print("Incorrect conversions:", test_currency_converter.log_incorrect_conversions)

    if final_value == expected_value:
        if not silent:
            print("Test successful!")
        return True
    else:
        if not silent:
            print("Test failed!")
        return False


def make_currencies_list_values(n_conversions: int) -> tuple[list[str], dict[str, float]]:
    currencies_list = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "CNY", "HKD", "DKR"]
    currencies_list = currencies_list + [el[::-1] for el in currencies_list]
    while len(currencies_list) < n_conversions + 1:
        new_currency = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
        if new_currency not in currencies_list:
            currencies_list.append(new_currency)
    currency_values = {
        "USD": 1.0,
    }
    for i in range(1, n_conversions + 1):
        currency_values[currencies_list[i]] = random.uniform(0.1, 2.0)
    return currencies_list, currency_values


def test_agent(is_code_agent: bool, n_conversions: int, step_by_step: bool, silent: bool = False):
    currencies_list, currency_values = make_currencies_list_values(n_conversions)

    currency_converter = CurrencyConverter(currency_values)

    step_handler = StepByStepHandler(
        step_by_step=step_by_step, n_iterations=n_conversions, currencies_list=currencies_list
    )

    # Create agent based on type
    verbosity_level = 0 if silent else 1
    if is_code_agent:
        agent = CodeAgent(
            [currency_converter],
            OpenAIServerModel(model_id="gpt-4o-mini"),
            step_callbacks=[step_handler.callback],
            max_steps=n_conversions * 2 + 2,
            verbosity_level=verbosity_level,
        )
    else:
        agent = ToolCallingAgent(
            [currency_converter],
            OpenAIServerModel(model_id="gpt-4o-mini"),
            step_callbacks=[step_handler.callback],
            max_steps=n_conversions * 2 + 2,
            verbosity_level=verbosity_level,
        )

    return test_agent_conversions(
        agent,
        n_conversions,
        currencies_list,
        currency_values,
        step_by_step=step_by_step,
        silent=silent,
    )


if __name__ == "__main__":
    args = parse_args()

    n_conversions = args.n_steps
    step_by_step = args.step_by_step
    is_code_agent = args.agent_type == "code"

    test_agent(is_code_agent, n_conversions, step_by_step)
