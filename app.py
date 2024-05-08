# Imports
from typing import List, Dict
import os
import json
import sys
import subprocess
import re

import warnings
warnings.filterwarnings('ignore')

from pydantic import BaseModel, Field, validator

# AI
import openai

# LangChain
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains  import LLMChain, ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

# Read environment: prepare for reading OPENAI_API_KEY
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# # # API KEY # # #
#openai.api_key  = os.getenv('OPENAI_API_KEY')
# openai.api_key = ''
# # # API KEY # # #

llm_model = "gpt-4-0613"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# ----------------------------
# Prompt Strings
# ----------------------------

user_prompt_str = """
Create a program that gets all employee first and last names whose salary average between 2018-2020 \
is above £25000 in the file ‘salaries.csv’.
"""

# Test case 1
"""
Create a program that reads and outputs the file ‘salaries.csv’.
"""

# Test case 2
"""
Create a program that gets the average salary of all employees between the years 2018-2020 in the file ‘salaries.csv’.
"""

# Test case 3
"""
Create a program that gets all employee first and last names whose salary average between 2018-2020 \
is above £25000 in the file ‘salaries.csv’.
"""

prog_spec_str = """
You are tasked with creating a program specification from a given user prompt.
A user's prompt will consist of the program's context (purpose) and specific requirements that accomplishes their goal.
The user's described program will be created using atomic functions in the Python 3.10 programming language.

The user's prompt is delimitered between ```.

```
{input}
```

From the user's prompt, identify the following:

Context:
- The purpose of the program and its intended functionality.


Specification:
- The key components, actions and requirements of the program.
- List the specific functionalities or features the program needs to accomplish.
- Identify any non-standard Python libraries required for the program.


Functions:
- The individual steps or functions required to fulfill the program's goal.
- Break down the tasks into manageable, atomic (single-purpose) units of functionality.
- Define inputs, processes, and outputs for each function.
- Ensure to define the function's name, list of parameters (name, datatype and context), \
    processes (specific tasks performed), and return value(s) (name, datatype and context).
- Datatypes should be defined as standard, e.g. instead of "string" use "str".
- You may use libraries that require installation/importing.
- Specifically for any csv or xml processes, use the Pandas library.
- You must include multiple tests that have a variety of correct, incorrect and erroneous data inputs.
- Test parameter values may use the return values of previous functions in the generated program's pipeline.


Parameters:
- List any specific parameters or inputs parsed to the program such as files, dates, etc.
- Determine how these parameters will be utilized within the program (their context).
- Clarify any constraints or limitations related to these parameters (their datatypes).


Result:
- The expected output or outcome of the program's execution.
- Define success criteria or metrics to evaluate the program's effectiveness.


Testing:
- Strategies for testing the program to ensure correctness and reliability.
- Define test cases and validation procedures to verify the program's functionality.
- You should include multiple tests that have a variety of correct, incorrect and erroneous data inputs.


{format_instructions}
"""

func_gen_str = '''
You are tasked with generating a Python code function snippet from a given function specification.
A function specification will consist of the function's name, context (purpose), parameters, processes, \
return values, and test cases (validation).
The function generated must be atomic and achieve the requirements defined in the function specification (scope).
The function generated will be using the Python (version 3.10) programming language.
Ensure that all variables (parameters, return values, etc) are given their relative datatypes when refering to them.
Utilise the test cases provided by the function specification to ensure validation is included within the code.

The function specification is delimited between ```.

```
{input}
```

{code_format_instructions}
'''

func_error_str = '''
You are tasked with debugging and bug-fixing a given Python code snippet when given the error encountered.
A code specification will consist of function name(s), context (purpose), parameters, processes, \
return values, and test cases (validation).
Functions in the Python code snippet must be atomic and achieve the requirements defined in the specification (scope).
The Python code snippet will be in the Python (version 3.10) programming language.
Ensure that all variables (parameters, return values, etc) are given their relative datatypes when refering to them.
Utilise the test cases provided by the specification to ensure validation is included within the code.

The code specification, Python code snippet and accompanying error is delimited between ```.

```
{input}
```

{code_format_instructions}
'''

code_format_instructions = '''
The output must be a markdown Python code snippet that includes the generated function's source code \
with leading and trailing "```python" and "```".
Only provide the markdown Python code snippet in your response.
Test cases inside the Python code snippet should be delimited by ###.
Within the code snippet, you must include a docstring inside the function using the following format:

"""
Context:
{function_context}

Parameters:
"{parameter_name}": {parameter_datatype} -> {parameter_context}

Returns:
"{return_name}": {return_datatype} -> {return_context}
"""

An example of a markdown code snippet is as follows:

```python
import pandas

def read_csv(file_path: str) -> p.DataFrame:
    """
    Context:
    This function is responsible for reading the contents of the CSV file.

    Parameters:
    "file_path": str -> The path to the CSV file that needs to be read.

    Returns:
    "data": pd.DataFrame -> The contents of the CSV file in a pandas DataFrame.
    """
    try:
        data: pd.DataFrame = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("The file does not exist.")
        return None

### Test cases
# The CSV file exists and can be read.
print(read_csv('salaries.csv'))

# The CSV file does not exist.
print(read_csv('non_existent.csv'))
###
```
'''

prog_gen_str = '''
You are tasked with creating a Python function that achieves the logical flow of a given program specification.
A program specification will consist of the program's context, function specifications and test cases (validation).
A function specification will consist of function name(s), context (purpose), parameters, processes, \
return values, and test cases (validation).
The Python code snippet will be in the Python (version 3.10) programming language.
You must create a "main()" function that links the program's functions together to achieve the goals defined in \
the program specificaiton.

The program specification and Python program code snippet will be delimited between ```.

```
{input}
```

{program_format_instructions}
'''

program_format_instructions = '''
The output must be a markdown Python code snippet that includes the generated function's source code \
with leading and trailing "```python" and "```".
Only provide the markdown Python code snippet for the "main()" function in your response \
alongside test cases exclusively, do not provide the entire program's code (only the generated "main()" function).
Test cases inside the Python code snippet should be delimited by ###.
Within the "main()" function code snippet, you must include a docstring inside the function using the following format:

"""
Context:
{function_context}

Parameters:
"{parameter_name}": {parameter_datatype} -> {parameter_context}

Returns:
"{return_name}": {return_datatype} -> {return_context}
"""

An example of a markdown code snippet is as follows:

```python
def main(file_path: str) -> float:
    """
    Context:
    The purpose of the program is to calculate the average salary of all employees \
    between the years 2018-2020 from a CSV file.

    Parameters:
    "file_path": str -> The path to the CSV file that needs to be read.

    Returns:
    "average_salary": float -> The average salary of all employees between the years 2018-2020.
    """
    data: pd.DataFrame = read_csv(file_path)
    
    if data is None:
        return None
        
    average_salary: float = calculate_average_salary(data)
    
    return average_salary

### Test cases
# The program is run with a valid CSV file.
print(main('salaries.csv'))

# The program is run with an invalid CSV file.
print(main('invalid.csv'))
###
```
'''

# ----------------------------
# Response Classes
# ----------------------------

# Response Classes using PydanticOutputParsers instead of using Response Schemas and StructuredOutputParsers
# This is better because it provides more control over the output structure, removing the responsibility from
# the LLM and instead providing it the structure to populate with its response.
# Also removes repeated instructions in the prompt format due to Response Schemas inserting into the prompt.

class Ret(BaseModel):
    return_name: str
    return_datatype: str
    return_context: str

class Param(BaseModel):
    parameter_name: str
    parameter_datatype: str
    parameter_context: str

class DocStr(BaseModel):
    docstring_context: str
    docstring_parameters: List[Param]
    docstring_returns: List[Ret]

class TestParam(BaseModel):
    parameter_name: str
    parameter_value: str
    parameter_datatype: str

class TestCase(BaseModel):
    testcase_context: str
    testcase_parameters: List[TestParam]
    testcase_result: str

class Func(BaseModel):
    function_name: str
    function_context: str
    function_parameters: List[Param]
    function_processes: List[str]
    function_returns: List[Ret]
    function_testcases: List[TestCase]

class Functions(BaseModel):
    functions: List[Func]

class Program(BaseModel):
    program_context: str
    program_specification: str
    program_functions: List[Func]
    program_parameters: List[Param]
    program_result: str
    program_testing: List[TestCase]

# ----------------------------
# Chains
# ----------------------------

"""
name:                   name of the chain.
description:            short description to help the LLM route between the chains.
prompt_string:          the string that contains the prompt template for the given chain.
response_class:         the main class that structures the LLM response (head).
messages:               dictionary containing any other prompt templates to be inserted into the main prompt string
                            other than the input.
format_instructions:    format generated using PydanticOutputParser.get_format_instructions() aka the JSON output.
"""

prompt_infos = [
    {
        "name": "program specification",
        "description": "Good for generating a program specification from a user prompt",
        "prompt_string": prog_spec_str,
        "response_class": Program,
        "messages": {
            "format_instructions": None
        }
    },
    {
        "name": "function generation",
        "description": "Good for generating code for a Python function from a given function specification",
        "prompt_string": func_gen_str,
        "response_class": None,
        "messages": {
            "code_format_instructions": code_format_instructions,
        }
    },
    {
        "name": "code debug",
        "description": "Good for debugging a code snippet with its accompanying encountered error",
        "prompt_string": func_error_str,
        "response_class": None,
        "messages": {
            "code_format_instructions": code_format_instructions
        }
    },
    {
        "name": "pipeline generation",
        "description": "Good for generating a program 'main()' function from a given program specification and code snippet",
        "prompt_string": prog_gen_str,
        "response_class": None,
        "messages": {
            "program_format_instructions": program_format_instructions
        }
    }
]

destination_chains = {}
for info in prompt_infos:
    # Check if response requires formatting from output schemas
    if info["response_class"]:
        output_parser = PydanticOutputParser(pydantic_object=info["response_class"])

        # JSON output formatting if required
        if "format_instructions" in info["messages"]:
            format_instructions = output_parser.get_format_instructions()
            info["messages"]["format_instructions"] = format_instructions

    # Create the template from the string, passing any extra inputs for the template as partial variables
    template = ChatPromptTemplate.from_template(
        template=info["prompt_string"],
        partial_variables=info["messages"]
    )

    chain = LLMChain(llm=llm, prompt=template)
    destination_chains[info["name"]] = chain


destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# ----------------------------
# Router Chain
# ----------------------------

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

# ----------------------------
# Program Specificaiton Chain Invoking
# ----------------------------

program_specification = chain.invoke(user_prompt_str)
print(program_specification["text"])

# Extract the functions from the response
functions_dict = json.loads(program_specification["text"])["program_functions"]

# ----------------------------
# Function Library Checking
# ----------------------------

# Here, the chain should go through each function spec and check whether the function exists
# within the function library, requiring Neo4j knowledge graph installation & integration (or alternatives).
# Essentially adding the 'RAG' part to the LLM.

# Not essential for now, get it working with the CORE FLOW (remove function library, generate on the fly).
# For now, this is skipped.

# Another optimisation would be to convert linear chaining for each function into asynchronous
# API calls (e.g. threading) for each function specification, speeding up the process of generation.

# ----------------------------
# Function Generation & Testing
# ----------------------------

def execute_function(code):
    # Write the code block to a new python environment
    with open("temp_code.py", "w") as f:
        f.write(code)

    # Execute the code block
    try:
        result = subprocess.run([sys.executable, "temp_code.py"], capture_output=True, text=True, check=True, timeout=10)
        return result.stdout, False
    except subprocess.CalledProcessError as e:
        return e.stdout + e.stderr, True
    except subprocess.TimeoutExpired:
        return "Timed out, code took too long to execute", True

def get_snippet(text, search_type="code"):
    if search_type == "code":
        pattern = r"```(?:\w+\s+)?(.*?)```"
    elif search_type == "test":
        pattern = r"###(?:\w+\s+)?(.*?)###"

    # Locate code in markdown ```python
    matches = re.findall(pattern, text, re.DOTALL)
    return [block.strip() for block in matches]

def generate_function(func):
    input_str = json.dumps(func)
    error = True

    while error:
        # Generate the code snippet for the given function
        response = chain.invoke(input_str)
        print(response["text"])

        # Get the markdown code block
        code_snippet = get_snippet(response["text"], "code")[0]

        # Create and execute the function
        output, error = execute_function(code_snippet)

        # Iterative development process goes here
        if error:
            print("Error in code\n", output)
            input_str = f"""
                Specification:
                {json.dumps(func)}

                Code Snippet:
                ```python
                {code_snippet}
                ```

                Error:
                {output}
                """
            print(input_str)

        else:
            print("No error\n", output)
            return response["text"]

def remove_test_cases_from_code(code):
    # Split the test cases from the rest of the code
    test_snippet = get_snippet(code, "test")[0]
    test_block = f"### {test_snippet}\n###"

    # Remove the test cases from the rest of the code
    code_block = code.replace(test_block, '')

    # Write snippet to program.py
    code_snippet = get_snippet(code_block, "code")[0] + "\n\n"

    return code_snippet

# Cleanup
open("program.py", "w").close()
program_code = ""

# For each function in the program specification
for func in functions_dict:
    # Generate the function
    gen_func = generate_function(func)

    code_snippet = remove_test_cases_from_code(gen_func)

    program_code += code_snippet
    with open("program.py", "a") as f:
        f.write(code_snippet)


input_str = f'''
Specification:
{json.dumps(program_specification["text"])}

Code Snippet:
{program_code}
'''

response = chain.invoke(input_str)
print(response["text"])

def execute_program(code):
    with open("program.py", "a") as f:
        f.write(code)

    try:
        result = subprocess.run([sys.executable, "program.py"], capture_output=True, text=True, check=True, timeout=20)
        return result.stdout, False
    except subprocess.CalledProcessError as e:
        return e.stdout + e.stderr, True
    except subprocess.TimeoutExpired:
        return "Timed out, code took too long to execute", True


code_snippet = get_snippet(response["text"], "code")[0] + "\n\n"
output, error = execute_program(code_snippet)

# code_snippet = remove_test_cases_from_code(response["text"])
# print(code_snippet)

