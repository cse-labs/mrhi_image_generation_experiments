import os
import importlib.util

RULES_DIR = 'rules'  # Update with the path to your rules/checks directory
README_PATH = 'README.md'  # Update with the path to your README.md

def get_all_rule_names():
    rule_files = [f[:-3] for f in os.listdir(RULES_DIR) if f.endswith('.py') and f != "__init__.py"]
    rule_names = []

    for rule_file in rule_files:
        spec = importlib.util.spec_from_file_location("module.name", os.path.join(RULES_DIR, rule_file + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rule_name = getattr(module, "RULE_NAME", None)
        if rule_name:
            rule_names.append(rule_name)

    return rule_names

def update_readme(rule_names):
    with open(README_PATH, 'r') as file:
        content = file.read()

    # Locate the "Currently Supported Checks" section
    start_marker = "# Currently Supported Checks"
    end_marker = "\n#"  # Any subsequent heading will start with '#'
    start_index = content.find(start_marker) + len(start_marker)
    
    # Find the next heading or use the end of the file
    end_index = content.find(end_marker, start_index)
    if end_index == -1:
        end_index = len(content)

    # Replace the old list with the new list
    new_content = content[:start_index] + "\n\n" + "\n".join("- " + rule for rule in rule_names) + "\n" + content[end_index:]

    with open(README_PATH, 'w') as file:
        file.write(new_content)

if __name__ == "__main__":
    rule_names = get_all_rule_names()
    update_readme(rule_names)
