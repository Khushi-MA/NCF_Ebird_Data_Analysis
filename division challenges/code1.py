import requests
from readability import Document
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import json
import pandas as pd
import re

def get_api_key():
    return "AIzaSyBMUGzMz7SxlBvJ2KdsQE0Ovez2kHJo3No"

def get_prompt_text():
    return (f"""
Extract structured information from the text below about a birding challenge. For each item, provide the value or list if available, or 'NONE' if missing.
Number of birders
Number of observations
Number of lists
Number of species
Number of unique lists with media
Names of birders (list)
Winner's name
How was the winner chosen
Location of the challenge
Number of checklist requirements for completing the challenge
Extra condition for completing the challenge 
Any tips or important points (list)
List of bird species mentioned (list)

Text:
\"\"\"{page_content}\"\"\"
""")

def extract_main_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except RequestException as e:
        return f"❌ Failed to fetch page: {e}"

    # Use Readability to extract the main article content
    doc = Document(response.text)
    html_content = doc.summary()
    title = doc.title()

    # Optionally clean with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)

    return f"📄 Title: {title}\n\n{text}"

def send_to_gemini(api_key: str, page_content: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json"
    }

    # Build the prompt with instructions + your scraped content
    prompt_text = get_prompt_text()

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt_text.strip()
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def parse_gemini_structured_response(text):
    """
    Parse Gemini's structured response into a dictionary.
    Assumes each attribute is on a new line, optionally with a colon.
    """
    fields = [
        "Number of birders",
        "Number of observations",
        "Number of lists",
        "Number of species",
        "Number of unique lists with media",
        "Names of birders",
        "Winner's name",
        "How was the winner chosen",
        "Location of the challenge",
        "Number of checklist requirements for completing the challenge",
        "Extra condition for completing the challenge",
        "Any tips or important points",
        "List of bird species mentioned"
    ]
    number_fields = [
        "Number of birders",
        "Number of observations",
        "Number of lists",
        "Number of species",
        "Number of unique lists with media",
        "Number of checklist requirements for completing the challenge"
    ]
    list_fields = [
        "Names of birders",
        "Any tips or important points",
        "List of bird species mentioned"
    ]
    text_fields = [
        "Winner's name",
        "How was the winner chosen",
        "Location of the challenge",
        "Extra condition for completing the challenge"
    ]

    result = {}
    for idx, field in enumerate(fields):
        next_field = fields[idx + 1] if idx + 1 < len(fields) else None
        if next_field:
            pattern = rf"{re.escape(field)}\s*[:\-]?\s*(.*?)(?=\n{re.escape(next_field)}\s*[:\-]?|\Z)"
        else:
            pattern = rf"{re.escape(field)}\s*[:\-]?\s*(.*)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if field in number_fields:
                num_match = re.search(r"[\d,]+", value)
                if num_match:
                    value = num_match.group(0).replace(",", "")
                else:
                    value = "NONE"
            elif field == "Names of birders":
                # Only keep lines that look like names (not field labels, not empty, not 'NONE')
                items = []
                for line in value.splitlines():
                    line = line.strip()
                    # Remove bullet/number and markdown
                    line = re.sub(r"^(\*|\-|\•|\d+\.)\s*", "", line)
                    line = re.sub(r"^\*\*|\*\*$", "", line)
                    line = line.strip()
                    # Filter: must contain at least one letter, not a field label, not 'NONE'
                    if (
                        line
                        and line.upper() != "NONE"
                        and not re.match(r"^(winner|how|location|number|extra|any tips|list of bird)", line, re.I)
                        and re.search(r"[A-Za-z]", line)
                    ):
                        items.append(line)
                value = items if items else "NONE"
            elif field in ["Any tips or important points", "List of bird species mentioned"]:
                items = []
                for line in value.splitlines():
                    line = line.strip()
                    if re.match(r"^(\*|\-|\•|\d+\.)\s+", line):
                        line = re.sub(r"^(\*|\-|\•|\d+\.)\s+", "", line)
                        line = re.sub(r"^\*\*|\*\*$", "", line)
                        line = line.strip()
                        if line and line.upper() != "NONE":
                            items.append(line)
                value = items if items else "NONE"
            elif field in text_fields:
                # Take the first non-empty line, remove markdown and field label
                lines = [l.strip() for l in value.splitlines() if l.strip()]
                if lines:
                    line = lines[0]
                    line = re.sub(r"^\*\*|\*\*$", "", line)
                    # Remove field label if present
                    line = re.sub(rf"^{re.escape(field)}\s*[:\-]?\s*", "", line, flags=re.I)
                    value = line.strip() if line.strip() else "NONE"
                else:
                    value = "NONE"
            else:
                value = value if value else "NONE"
            result[field] = value
        else:
            result[field] = "NONE"
    return result

def pretty_print_gemini_response(response_json):
    try:
        # Extract the first candidate's first part's text
        candidates = response_json.get("candidates", [])
        if not candidates:
            print("No candidates found in response.")
            return

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            print("No parts found in content.")
            return

        text = parts[0].get("text", "")

        # Print clean output
        print("\n=== Extracted Structured Information ===\n")
        print(text.strip())

    except Exception as e:
        print(f"Error while parsing response: {e}")


    try:
        # Extract the first candidate's first part's text
        candidates = response_json.get("candidates", [])
        if not candidates:
            print("No candidates found in response.")
            return

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            print("No parts found in content.")
            return

        text = parts[0].get("text", "")

        # Print clean output
        print("\n=== Extracted Structured Information ===\n")
        print(text.strip())

    except Exception as e:
        print(f"Error while parsing response: {e}")

file_name = "ebird_challenges_all.xlsx"



if __name__ == "__main__":
    df = pd.read_excel(file_name)
    urls = df["Article URL"].dropna().tolist()
    API_KEY = get_api_key()

    # Prepare columns for all attributes if not already present
    gemini_fields = [
        "Number of birders",
        "Number of observations",
        "Number of lists",
        "Number of species",
        "Number of unique lists with media",
        "Names of birders",
        "Winner's name",
        "How was the winner chosen",
        "Location of the challenge",
        "Number of checklist requirements for completing the challenge",
        "Extra condition for completing the challenge",
        "Any tips or important points",
        "List of bird species mentioned"
    ]
    for field in gemini_fields:
        if field not in df.columns:
            df[field] = None

    for idx, url in enumerate(urls, start=1):
        print(f"\n🔗 Processing URL {idx}: {url}")
        try:
            page_content = extract_main_content(url)
            response = send_to_gemini(API_KEY, page_content)
            # Extract text from Gemini response
            candidates = response.get("candidates", [])
            if not candidates:
                print("No candidates found in response.")
                continue
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                print("No parts found in content.")
                continue
            text = parts[0].get("text", "")
            # Parse structured response
            parsed = parse_gemini_structured_response(text)
            # Update DataFrame row for this URL
            row_idx = df.index[df["Article URL"] == url][0]
            for field in gemini_fields:
                value = parsed[field]
                # Store lists as JSON strings for Excel compatibility
                if isinstance(value, list):
                    value = json.dumps(value, ensure_ascii=False)
                df.at[row_idx, field] = value
            print("📥 Gemini Response:")
            pretty_print_gemini_response(response)
        except Exception as e:
            print(f"❌ Failed to process URL {idx}: {e}")

    # Save updated DataFrame back to Excel
    df.to_excel("file1.xlsx", index=False)