import pandas as pd
import json
import requests
from readability import Document
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import numpy as np
import sys
import os

FILE = "sample3.csv"  # Path to your CSV file with URLs

API_KEYS = [
    "AIzaSyBMUGzMz7SxlBvJ2KdsQE0Ovez2kHJo3No",  # your current key
    "AIzaSyBgPeDlSd3Ddh_aR1jwhbnbtjv7eqFdYfc",
    "AIzaSyDpo9g04Qc8HIEPdN_Zt5WwnkwMkdpu0T4",
    "AIzaSyBSX0pzvlMY9aFb4tRoif78EUvYz63us8w",
    "AIzaSyDYHssjIfqlZyaiGzbB3cyVW-dpL-xPzQI",
    "AIzaSyBLH6nM_fRjRzwa44q_8XxhDCzoW6-pXOE"
]

PROMPT_TEXT = """
Extract structured information from the text below about a birding challenge. 
Return your answer as a JSON object with the following keys:
- number_of_birders (integer or "Not found")
- number_of_observations (integer or "Not found")
- number_of_lists (integer or "Not found")
- number_of_species (integer or "Not found")
- number_of_unique_lists_with_media (integer or "Not found")
- names_of_birders (list of strings or "Not found")
- winner_name (string or "Not found")
- how_was_winner_chosen (string or "Not found")
- location_of_challenge (string or "Not found")
- checklist_requirements (string or "Not found")
- extra_condition (string or "Not found")
- tips_or_important_points (list of strings or "Not found")
- bird_species_mentioned (list of strings or "Not found")

If a value is missing, use "Not found". Do not include any explanation or markdown, only output the JSON.

Text:
\"\"\"{page_content}\"\"\"
"""

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
    prompt_text = PROMPT_TEXT.format(page_content=page_content)

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

# def pretty_print_gemini_response(response_json):
#     try:
#         # Extract the first candidate's first part's text
#         candidates = response_json.get("candidates", [])
#         if not candidates:
#             print("No candidates found in response.")
#             return

#         content = candidates[0].get("content", {})
#         parts = content.get("parts", [])
#         if not parts:
#             print("No parts found in content.")
#             return

#         text = parts[0].get("text", "")

#         # Print clean output
#         print("\n=== Extracted Structured Information ===\n")
#         print(text.strip())

#     except Exception as e:
#         print(f"Error while parsing response: {e}")

def extract_clean_gemini_json(response_json, not_found_as_null=True):
    """
    Extracts and cleans the JSON object from Gemini's response, returning only the required fields.
    If not_found_as_null is True, replaces 'Not found' with ''.
    """
    import json
    required_keys = [
        "number_of_birders",
        "number_of_observations",
        "number_of_lists",
        "number_of_species",
        "number_of_unique_lists_with_media",
        "names_of_birders",
        "winner_name",
        "how_was_winner_chosen",
        "location_of_challenge",
        "checklist_requirements",
        "extra_condition",
        "tips_or_important_points",
        "bird_species_mentioned"
    ]
    # Extract the text from Gemini response
    candidates = response_json.get("candidates", [])
    if not candidates:
        return {k: ("" if not_found_as_null else "Not found") for k in required_keys}
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not parts:
        return {k: ("" if not_found_as_null else "Not found") for k in required_keys}
    text = parts[0].get("text", "")
    # Remove markdown code block if present
    if text.strip().startswith("```json"):
        text = text.strip().lstrip("`json").strip('`').strip()
        text = text[text.find('{'):]  # start from first {
    elif text.strip().startswith("```"):
        text = text.strip().lstrip("`").strip()
        text = text[text.find('{'):]
    # Try to load JSON
    try:
        data = json.loads(text)
    except Exception:
        # Try to extract JSON substring
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = {}
        else:
            data = {}
    # Build clean dict with only required keys
    clean = {}
    for k in required_keys:
        v = data.get(k, "Not found")
        if not_found_as_null and v == "Not found":
            v = ""
        clean[k] = v
    return clean

def ensure_gemini_columns_exist(df):
    gemini_columns = [
        "number_of_birders",
        "number_of_observations",
        "number_of_lists",
        "number_of_species",
        "number_of_unique_lists_with_media",
        "names_of_birders",
        "winner_name",
        "how_was_winner_chosen",
        "location_of_challenge",
        "checklist_requirements",
        "extra_condition",
        "tips_or_important_points",
        "bird_species_mentioned"
    ]
    for col in gemini_columns:
        if col not in df.columns:
            df[col] = ""
        # Always convert to object dtype
        df[col] = df[col].astype(object)
    return df

def append_gemini_response_to_df_row(df, url, gemini_response):
    """
    Appends Gemini response values to the DataFrame row corresponding to the given URL.
    Lists are saved as JSON strings. Handles dtype compatibility for numeric columns.
    """
    import json
    row_idx = df.index[df["Article URL"] == url]
    if len(row_idx) == 0:
        return df  # URL not found, do nothing
    row_idx = row_idx[0]
    for key, value in gemini_response.items():
        # Save lists as JSON strings
        if isinstance(value, list):
            value = json.dumps(value, ensure_ascii=False)
        # Handle missing values for numeric columns
        col_dtype = df[key].dtype if key in df.columns else None
        if value is None or value == "":
            if col_dtype is not None and (pd.api.types.is_float_dtype(col_dtype) or pd.api.types.is_integer_dtype(col_dtype)):
                value = np.nan
            else:
                value = ""
        df.at[row_idx, key] = value
    return df

def append_values_from_url(df, url, api_keys, max_attempts=10):
    """
    For a given URL, fetches content, gets Gemini response using rotating API keys in a round-robin fashion.
    If all keys fail in a full cycle, stops the system.
    Returns the updated DataFrame.
    """
    import time
    page_content = extract_main_content(url)
    attempts = 0
    key_count = len(api_keys)
    while attempts < max_attempts:
        api_key = api_keys[attempts % key_count]
        try:
            response = send_to_gemini(api_key, page_content)
            clean_response = extract_clean_gemini_json(response)
            # print(f"📥 Gemini Response (API {api_key}):", clean_response)
            df = append_gemini_response_to_df_row(df, url, clean_response)
            return df  # Success, return updated df
        except Exception as e:
            print(f"❌ API key {api_key} failed for URL {url}: {e}")
            attempts += 1
            if attempts % key_count == 0:
                print(f"⚠️ All API keys failed for URL {url} in this cycle. Retrying...")
            time.sleep(1)  # Optional: avoid hammering the API
    print(f"❌ All API keys failed for URL {url} after {max_attempts} attempts. Stopping system.")
    raise RuntimeError(f"All API keys failed for URL {url} after {max_attempts} attempts.")

def safe_read_csv_file(file):
    try:
        return pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"⚠️ UTF-8 decode failed for '{file}', trying latin1 encoding...")
        try:
            return pd.read_csv(file, encoding='latin1')
        except Exception as e:
            print(f"❌ Could not open file '{file}' for reading: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Could not open file '{file}' for reading: {e}")
        sys.exit(1)

def safe_write_csv_file(df, file):
    try:
        if os.name == 'nt':  # Windows
            import msvcrt
            try:
                with open(file, 'r+b') as f:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError as e:
                print(f"❌ File '{file}' is locked (possibly open in Excel): {e}")
                sys.exit(1)
        else:
            # On Unix, try to open for writing
            with open(file, 'ab') as f:
                pass
        df.to_csv(file, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"❌ Could not open file '{file}' for writing: {e}")
        sys.exit(1)

def update_csv_row_by_index(file, row_idx, row_data):
    import csv
    import numpy as np
    # Read all lines
    with open(file, 'r', encoding='utf-8', newline='') as f:
        lines = list(csv.reader(f))
    header = lines[0]
    # Convert np.nan to empty string for each value
    def safe_str(val):
        if pd.isna(val):
            return ''
        return str(val)
    lines[row_idx+1] = [safe_str(row_data.get(col, '')) for col in header]
    # Write all lines back
    with open(file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(lines)

if __name__ == "__main__":
    df = safe_read_csv_file(FILE)
    urls = df["Article URL"].dropna().tolist()
    # urls = urls[:4]  # Remove or adjust this for full run

    df = ensure_gemini_columns_exist(df)

    for idx, url in enumerate(urls, start=1):
        print(f"\n🔗 Processing URL {idx}: {url}")
        df = append_values_from_url(df, url, API_KEYS)
        print(f"✅ Updated row for URL: {url}")

    safe_write_csv_file(df, FILE)
    print("\n✅ All URLs processed successfully. Saving updated DataFrame to CSV.")