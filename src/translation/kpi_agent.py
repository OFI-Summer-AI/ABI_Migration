import os
import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Pydantic models for structured output
class TranslatedKPI(BaseModel):
    kpi_id: str
    name: str
    final_sql: str = Field(
        description="The fully resolved standard SQL formula. All KPI(\"...\") references must be replaced with their actual underlying SQL logic."
    )
    english_description: str = Field(
        description="A simple, plain English description of what this KPI or attribute measures."
    )

# List of translated KPIs
class TranslatedKPIList(BaseModel):
    kpis: List[TranslatedKPI]

# Translate KPIs
def translate_kpis(input_filepath: str, output_filepath: str):
    print(f"Loading KPIs from {input_filepath}...")
    with open(input_filepath, 'r', encoding='utf-8') as f:
        kpi_data = json.load(f)

    # Initialize the Groq LLM
    # We use a robust model like llama-3.3-70b-versatile or llama3-70b-8192
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables (.env). Please provide it.")
        
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=api_key
    )

    # Create a structured output parser using the Pydantic model
    structured_llm = llm.with_structured_output(TranslatedKPIList)

    system_prompt = """You are an expert Data Engineer and SQL Developer.
I will give you a JSON array of extracted KPIs and attributes. 
Some of the 'pql_formula' fields use the syntax KPI("kpi_id") to reference another KPI.

Your task is to:
1. Understand the dependencies between these KPIs.
2. Convert the 'pql_formula' to standard SQL.
3. Resolve all dependencies: replace any KPI("...") reference with the fully expanded SQL logic of that referenced KPI.
4. Write a simple, plain English description for each KPI or attribute explaining what it measures.

Output the result as a list of translated KPIs using the provided schema.
Ensure the SQL is accurate and ready to be executed against a standard SQL database.
"""

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Here are the extracted KPIs:\n\n{kpis}")
    ])

    # Create a chain
    chain = prompt | structured_llm

    print("Sending data to Groq LLM for translation and dependency resolution...")
    # Convert data back to JSON string for the prompt to keep it readable
    kpis_json_str = json.dumps(kpi_data, indent=2)
    
    # Run the chain
    result = chain.invoke({"kpis": kpis_json_str})

    # Prepare output directory
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the processed data
    output_data = [kpi.model_dump() for kpi in result.kpis]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully processed {len(output_data)} KPIs.")
    print(f"Results saved to {output_filepath}")

if __name__ == "__main__":
    # Define paths relative to the project root
    base_dir = Path(__file__).parent.parent.parent
    input_file = base_dir / "data" / "raw" / "extracted_kpis.json"
    output_file = base_dir / "data" / "processed" / "translated_kpis.json"
    
    # Run the translation
    import traceback
    try:
        translate_kpis(str(input_file), str(output_file))
    except Exception as e:
        with open("error.txt", "w") as f:
            f.write(f"Error during execution: {e}\n")
            traceback.print_exc(file=f)
