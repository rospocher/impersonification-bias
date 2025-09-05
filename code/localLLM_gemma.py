import torch

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os
import re

import sys

if len(sys.argv) > 3:
    variant = sys.argv[1]
    TEMPERATURE = float(sys.argv[2])
    NUMBER_OF_ITERATIONS = int(sys.argv[3])
    print(f"The parameter passed is: {variant}, {TEMPERATURE}, {NUMBER_OF_ITERATIONS}")
else:
    variant = "_generic"
    TEMPERATURE=1.0
    NUMBER_OF_ITERATIONS = 100
    print("No parameter was passed. Defaulting to: {variant}, {TEMPERATURE}, {NUMBER_OF_ITERATIONS}")
    exit

# 1. Choose a GGUF model from Hugging Face Hub
#model_id = "allenai/OLMo-2-0325-32B-Instruct-GGUF"
#model_basename = "OLMo-2-0325-32B-Instruct-Q5_K_S.gguf" # Choose a quantization that fits your VRAM/RAM

model_id = "google/gemma-3-27b-it-qat-q4_0-gguf"
model_basename = "gemma-3-27b-it-q4_0.gguf" # This is the specific GGUF file name


# 2. Download the GGUF model
model_path = hf_hub_download(
    repo_id=model_id,
    filename=model_basename,
    resume_download=True
)

print(f"Model downloaded to: {model_path}")


# 2. Download the GGUF model
model_path = hf_hub_download(
    repo_id=model_id,
    filename=model_basename,
    resume_download=True
)

print(f"Model downloaded to: {model_path}")

# 3. Load the model using llama_cpp.Llama
# Key parameters for memory management:
#   n_gpu_layers: Number of layers to offload to the GPU.
#                 Set to -1 to offload all layers possible.
#                 Set to 0 to run entirely on CPU.
#   n_ctx: Context window size (how many tokens the model can "remember").
#          Larger values require more memory.
#   n_batch: How many tokens are processed in parallel during generation.
#            Larger values can speed up generation but use more VRAM.

# Adjust n_gpu_layers based on your GPU VRAM:
# For a 7B model Q4_K_M, you might need ~4.5GB VRAM.
# You'll need to experiment with this value for your specific GPU.
# If you have 8GB VRAM, you might try 20, 30, or even -1 (all).
# If you only have system RAM, set n_gpu_layers=0
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1, # <--- This is where you pass the parameter
    n_ctx=2048,
    n_batch=512,
    #verbose=True     # <--- IMPORTANT: This will print messages about GPU offloading
)
print("\nModel loading complete. Please review the output above this line for details on GPU offloading.")
print("Look for lines like 'llm_load_tensors: offloading X/Y layers to GPU'.")
print("If no such lines appear, or if X is 0, then no layers were offloaded to the GPU (or you set n_gpu_layers=0).")


# Load prompt
with open('final_prompt_revised'+variant+'.txt', 'r', encoding='utf-8') as text:
    promptRaw = text.read()
    #print(promptRaw)

with open('questionnaire.txt', 'r', encoding='utf-8') as text:
    questionnaireRaw = text.read()
    #print(questionnaireRaw)

# Adapted prompt for Gemma models
prompt = f"""<start_of_turn>user
{promptRaw}

{questionnaireRaw}<end_of_turn>
<start_of_turn>model
"""

import json
import csv
from io import StringIO

def json_questionairre_to_dict(questionnaireRaw):
    questionsData = json.loads(questionnaireRaw)["questionnaire"]
    questions = dict()
    for question in questionsData:
        id = question["id"]
        options = question["options"]
        questions[id]=options
    return questions


def clean_json_string(raw_text):
    """
    Cleans a JSON string by removing markdown code block formatting (e.g. ```json ... ```).
    """
    cleaned = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', raw_text.strip(), flags=re.IGNORECASE)
    return cleaned


def json_to_csv_single_row_semicolon(json_string, questions):
    """
    Converts a JSON string (representing a single set of survey responses)
    into a semicolon-separated CSV string with only the values, ordered by question ID.

    Args:
        json_string (str): The JSON data as a string.
        questions (dict): The dictionary containing question IDs and their valid options.

    Returns:
        str: The semicolon-separated CSV data as a string, or an error message.
    """
    cleaned_json = clean_json_string(json_string)

    try:
        data = json.loads(cleaned_json)    
    except json.JSONDecodeError:
        print(f"Failed to decode JSON: {json_string}")
        return "JSON_DECODE_ERROR"


    # Determine the number of questions based on your questionnaire
    num_questions = 13
    ordered_values = []

    for i in range(1, num_questions + 1):
        key = f"q{i}".strip()
        value = data[key].strip()
        if key not in data:
            print(f"Missing expected key: {key} in the JSON data.")
            return f"MISSING KEY {key}"
        if value not in questions[key]:
            print(f"Returned not valid option: {value} in the JSON data.")
            return f"WRONG VALUE {value} for {key}"           
        ordered_values.append(data[key])
        
    output = StringIO()
    # Specify the delimiter as ';' for semicolon separation
    writer = csv.writer(output, delimiter=';')
    writer.writerow(ordered_values)

    return output.getvalue()
    
questions = json_questionairre_to_dict(questionnaireRaw)
print(questions)

with open('log'+variant+'_'+model_id.replace("/","_")+'_'+str(NUMBER_OF_ITERATIONS)+'_T'+str(TEMPERATURE)+'.log', 'w', encoding='utf-8') as log:
    with open('responses'+variant+'_'+model_id.replace("/","_")+'_'+str(NUMBER_OF_ITERATIONS)+'_T'+str(TEMPERATURE)+'.csv', 'w', encoding='utf-8') as file:

        # chat = client.chats.create(model=MODEL_ID, config=MODEL_CONFIG)  # one chat for all iterations
        file.write("age;sex;gender;marital_status;citizenship;ethnicity;educational_attainment;employment_status;industry;hours;earnings;disability_status;life_satisfaction\n")
        i = 0
        while i < NUMBER_OF_ITERATIONS:
            print("ITERATION:",i)
            log.write("ITERATION: " + str(i) + "\n")

            output = llm(
                prompt,
                max_tokens=1000,
                temperature=TEMPERATURE,
                top_p=0.95,
                stop=["###"],
                echo=False,
                stream=False
            )
            
            generated = output["choices"][0]["text"].strip()
            log.write(generated + "\n")
            text = json_to_csv_single_row_semicolon(generated, questions)
            #print(csv_output)
            if text.startswith("MISSING KEY"):
                log.write(f'MISSING KEY in {generated}\n')
            elif text.startswith("WRONG VALUE"):
                log.write(f'WRONG VALUE in {generated}\n')
            else:
                i += 1
                file.write(text) 
