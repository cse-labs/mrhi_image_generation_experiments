import json
import os
import importlib
from tabulate import tabulate
from colorama import init, Fore
from dotenv import load_dotenv


init(autoreset=True)
# Load environment variables from .env file
load_dotenv()

def load_rules():
    """Dynamically load rules from the rules directory."""
    rule_files = [f[:-3] for f in os.listdir(os.path.join(os.path.dirname(__file__),
                "rules")) if f.endswith(".py") and f != "__init__.py"]
    loaded_rules = {}
    for rule_file in rule_files:
        module = importlib.import_module(f"mrhivalidator.rules.{rule_file}")
        rule_name = getattr(module, "RULE_NAME", None)  # Get the rule's logical name
        if not rule_name:
            print(f"Warning: No RULE_NAME found in {rule_file}. Skipping.")
            continue
        rule_class = getattr(module, rule_name, None)
        if not rule_class:
            print(f"Warning: No rule class found for {rule_name}. Skipping.")
            continue
        rule_instance = rule_class()  # Instantiate the rule class
        loaded_rules[rule_name] = rule_instance
    return loaded_rules

def apply_rules_to_image(active_rules_config, loaded_rules):
    results = {}

    for rule_name, rule_configs in active_rules_config.items():
        for index, rule_config in enumerate(rule_configs):
            rule = loaded_rules.get(rule_name)
            if not rule:
                print(f"Warning: {rule_name} not found!")
                continue
            
            kwargs = rule_config.get('kwargs', {})

            try:
                # Expect rule to return an instance of EvaluationResult
                result = rule.evaluate(**kwargs)
                status = result.result
                value = result.score
                rng = result.range
                explanation = result.explanation  
                unique_key = f"{rule_name}_{index}" # This ensures uniqueness for each configuration
                results[unique_key] = {"status": status.name, "value": value, "range": rng, "explanation": explanation}
            
            except Exception as e:
                print(f"Error evaluating rule {rule_name}: {str(e)}")
                results[rule_name] = {"status": "ERROR", "value": None, "range": None ,"explanation": None}

    return results



def validate(configs=None,table_output=True):

    if not configs:
        config_file_path = "../configurations.json"
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as file:
                configs = json.load(file)
        else:
            raise ValueError("No configuration file found and no configs provided.")

    # Dynamically load rules 
    loaded_rules = load_rules()

    # Apply rules to the image
    rule_results = apply_rules_to_image(configs["active_rules"], loaded_rules)

    if not table_output:
        return rule_results

    # Display results in tabular format
    table_data = []
    for rule, result in rule_results.items():
        color = Fore.GREEN if result["status"] == "PASS" else Fore.RED
        table_data.append([rule, color + result["status"], result["value"], result["range"],result["explanation"]])
    print(tabulate(table_data, headers=['Rule', 'Status', 'Value', 'Range','Explanation']))

if __name__ == "__main__":
    validate()