import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from smolagents import ActionStep, CodeAgent, MultiStepAgent, OpenAIServerModel, Tool, ToolCallingAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-conversions", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--agent-type", type=str, default="code")
    parser.add_argument("--max-workers", type=int, default=64)
    return parser.parse_args()


class AgentFailedError(ValueError):
    def __init__(self, step, *args: object) -> None:
        super().__init__(*args)
        self.step = step


class StepByStepHandler:
    def __init__(
        self,
        step_by_step: bool,
        n_iterations: int,
        currencies_list: list[str],
        expected_amounts: list[float],
        used_tool: Tool,
    ):
        self.step_by_step = step_by_step
        self.current_step = 1
        self.log_incorrect_conversions: list = []
        self.n_iterations = n_iterations
        self.currencies_list = currencies_list
        self.used_tool = used_tool
        self.expected_amounts = expected_amounts

    def callback(self, memory_step: ActionStep):
        last_triplet = self.used_tool.last_conversion_triplet  # type: ignore

        # Chekc correctness and advancement
        if self.current_step < self.n_iterations:
            if (
                last_triplet[1] == self.currencies_list[self.current_step]
                and last_triplet[2] == self.currencies_list[self.current_step + 1]
            ):
                # Check if the amount used was corect
                if last_triplet[0] != self.expected_amounts[self.current_step]:
                    raise AgentFailedError(step=self.current_step)
                else:
                    # else, it's good to go
                    # Update the current step!
                    self.current_step += 1
        # Provide guidance on next step
        if self.current_step < self.n_iterations:
            guidance = f"\n\n-> Once expressed in {self.currencies_list[self.current_step]}, the amount should be converted to {self.currencies_list[self.current_step + 1]} => Do this now, don't do anything else!"
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
        self.current_step = 0
        self.last_conversion_triplet: tuple | None = None

    def forward(self, amount: float, from_currency: str, to_currency: str) -> float:
        self.conversions_done.append((amount, from_currency, to_currency))
        calculated_amount = (
            amount * self.currency_values[to_currency] / self.currency_values[from_currency] * (1 + self.tax)
        )
        self.last_conversion_triplet = (amount, from_currency, to_currency)
        return calculated_amount


def make_task(
    original_amount: float,
    agent: MultiStepAgent,
    n_conversions: int,
    currencies_list: list[str],
    step_by_step: bool = False,
) -> str:
    agent_specific_instructions = (
        ("Return the number as a string, with only the number, no other text, like this: '1329953422.2198896'.")
        if isinstance(agent, ToolCallingAgent)
        else ""
    )

    if step_by_step:
        task = f"""You have been given an amount of money of {original_amount} USD.
You will have to successively have it pass through {n_conversions} countries as an items travels, each time converting money to the local currency and applying tax. Some of these currencies might be fictional.
You have got access to the tool currency_converter, which will help you convert the amount of money and apply tax to handle the transfer from one currency to another.
This tool will also give you updated instructions in the logs on what conversion to do next. So whenever you don't know what to do next, just look at the logs from the previous tool use!
{agent_specific_instructions}

Here is the first conversion you will have to do:
Convert from {currencies_list[0]} to {currencies_list[1]}
"""
    else:
        task = f"""You have been given an amount of money of {original_amount} USD.
You will have to successively have it pass through {n_conversions} countries as an items travels, each time converting money to the local currency and applying tax. Some of these currencies might be fictional.
You have got access to the tool currency_converter, which will help you convert the amount of money and apply tax to handle the transfer from one currency to another.
{agent_specific_instructions}

Here is the list of conversions you have to do:
"""
        for i in range(n_conversions):
            task += f"Convert from {currencies_list[i]} to {currencies_list[i + 1]}\n"

    task += "Now go on!"

    return task


def make_currencies_list_values(n_conversions: int) -> tuple[list[str], dict[str, float]]:
    currencies_list = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "CNY", "HKD", "DKK"]
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

    test_currency_converter = CurrencyConverter(currency_values)

    original_amount = random.uniform(100.0, 1000.0)

    expected_amounts = [original_amount]
    for i in range(n_conversions):
        if not silent:
            print(f"Amount: {expected_amounts[-1]}, converting from {currencies_list[i]} to {currencies_list[i + 1]}")
        expected_amounts.append(
            test_currency_converter.forward(expected_amounts[-1], currencies_list[i], currencies_list[i + 1])
        )

    step_handler = StepByStepHandler(
        step_by_step=step_by_step,
        n_iterations=n_conversions,
        currencies_list=currencies_list,
        expected_amounts=expected_amounts,
        used_tool=currency_converter,
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

    task = make_task(
        original_amount,
        agent,
        n_conversions,
        currencies_list,
        step_by_step,
    )
    try:
        final_amount = agent.run(task)
    except AgentFailedError as error:
        step_failure = error.step
        print(f"Failed at step {step_failure} out of {n_conversions}")
        return False

    try:
        final_amount = float(str(final_amount).replace("'", "").replace('"', ""))  # type: ignore
    except ValueError:
        print("Output is not a number:", final_amount)
        return False
    if not silent:
        print("Final value:", final_amount)
        print("Expected value:", expected_amounts[-1])

    if final_amount == expected_amounts[-1]:
        if not silent:
            print("Test successful!")
        return True
    else:
        if not silent:
            print("Test failed!")
        return False


def wrap_test_agent(is_code_agent: bool, n_conversions: int, step_by_step: bool, i: int):
    return {f"{n_conversions}_{i}": test_agent(is_code_agent, n_conversions, step_by_step, silent=True)}


if __name__ == "__main__":
    args = parse_args()
    print(f"Starting run with arguments: {args}")

    is_code_agent = args.agent_type == "code"
    step_by_step = True

    with ThreadPoolExecutor(max_workers=args.max_workers) as exe:
        results: dict[int, list[bool]] = {}
        futures: list = []
        for n_conversions in range(10, args.n_conversions + 1, 10):
            results[n_conversions] = []
            for i in range(args.n_samples):
                futures.append(exe.submit(wrap_test_agent, is_code_agent, n_conversions, step_by_step, i))

        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
            outcomes = f.result()
            for name, result in outcomes.items():
                n_conversions = int(name.split("_")[0])
                results[n_conversions].append(result if result is not None else False)

    with open(f"benchmark_results_{args.agent_type}.json", "w") as f:
        json.dump(results, f)

    print("All tasks processed.")
