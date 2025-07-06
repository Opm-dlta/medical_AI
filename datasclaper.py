import json
import requests
from bs4 import BeautifulSoup
import time
import os

# Updated Medical condition URLs organized by category and source
URLS = {
    # General Health & Common Illnesses
    "Everyday Health Common Illnesses": "https://www.everydayhealth.com/conditions/",
    "Healthdirect Australia Common Questions": "https://www.healthdirect.gov.au/common-health-questions",
    "Better Health Channel Conditions": "https://www.betterhealth.vic.gov.au/health/conditionsandtreatments",
    
    # Infectious Illnesses & Self-Care
    "NHS Inform Self-Help Guides": "https://www.nhsinform.scot/self-help-guides",
    "Healthline Cold and Flu": "https://www.healthline.com/health/cold-flu",
    "Cleveland Clinic Cold vs Flu": "https://health.clevelandclinic.org/cold-vs-flu-vs-covid-19/",
    
    # Childhood Illnesses
    "KidsHealth Illnesses and Injuries": "https://kidshealth.org/en/parents/illnesses-injuries/",
    "HealthyChildren Common Conditions": "https://www.healthychildren.org/English/health-issues/conditions/Pages/default.aspx",
    
    # Skin, Ear, and Eye Conditions
    "AAD Skin Conditions": "https://www.aad.org/public/diseases",
    "AAO Eye Health Topics": "https://www.aao.org/eye-health",
    "ENT Health Conditions": "https://www.enthealth.org/conditions/",
    
    # Digestive & Urinary Health
    "NIDDK Digestive Diseases": "https://www.niddk.nih.gov/health-information/digestive-diseases",
    "Cleveland Clinic UTIs": "https://my.clevelandclinic.org/health/diseases/15658-urinary-tract-infection-uti"
}

def check_url_status(url):
    """Check if URL is accessible and return status info"""
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        return {
            'accessible': response.status_code == 200,
            'status_code': response.status_code,
            'error': None
        }
    except requests.exceptions.RequestException as e:
        return {
            'accessible': False,
            'status_code': None,
            'error': str(e)
        }

def scrape_sections(url):
    """Scrape sections from URL with error handling"""
    try:
        # First check if URL is accessible
        status_info = check_url_status(url)
        if not status_info['accessible']:
            raise Exception(f"URL not accessible: Status {status_info['status_code']}, Error: {status_info['error']}")
        
        # Make the actual request
        r = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Check for HTTP errors
        if r.status_code != 200:
            raise Exception(f"HTTP {r.status_code} error")
        
        # Check if content is actually HTML
        content_type = r.headers.get('content-type', '').lower()
        if 'text/html' not in content_type and 'text/plain' not in content_type:
            raise Exception(f"Content is not HTML: {content_type}")
        
        soup = BeautifulSoup(r.text, "html.parser")

        # Get title
        title = soup.find("h1")
        title = title.get_text(strip=True) if title else "Condition"

        def get_block(heading_texts):
            """Extract content blocks based on heading text"""
            for h in soup.find_all(["h1", "h2", "h3", "h4", "strong", "b"]):
                h_text = h.get_text(strip=True).lower()
                if any(txt.lower() in h_text for txt in heading_texts):
                    out = []
                    for sib in h.find_next_siblings():
                        if sib.name in ["h1", "h2", "h3", "h4", "strong", "b"]:
                            break
                        if sib.name == "p":
                            text = sib.get_text(strip=True)
                            if text:
                                out.append(text)
                        elif sib.name == "ul":
                            for li in sib.find_all("li"):
                                text = li.get_text(strip=True)
                                if text:
                                    out.append(f"• {text}")
                    return "\n".join(out)
            return ""

        # Try to extract different sections
        overview = get_block(["overview", "about", "what is", "definition", "description"])
        symptoms = get_block(["symptoms", "signs", "warning signs", "when to seek"])
        treatment = get_block(["treatment", "first aid", "what to do", "steps", "action"])
        medication = get_block(["medication", "drugs", "medicines", "treatment options"])
        when_to_seek = get_block(["seek help", "when to call", "emergency", "911", "doctor"])

        # If we can't find specific sections, try to get general content
        if not any([overview, symptoms, treatment, medication, when_to_seek]):
            # Get all paragraphs as fallback
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            if paragraphs:
                overview = "\n".join(paragraphs[:3])  # First 3 paragraphs
            else:
                raise Exception("No useful content found on page")

        return title, overview, symptoms, treatment, medication, when_to_seek

    except requests.exceptions.Timeout:
        raise Exception("Request timeout")
    except requests.exceptions.ConnectionError:
        raise Exception("Connection error")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request error: {e}")
    except Exception as e:
        raise Exception(f"Scraping error: {e}")

def convert_json_to_jsonl(json_path, out_jsonl_path, append=False):
    """Convert JSON data to JSONL format with option to append to existing file"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    mode = "a" if append else "w"
    with open(out_jsonl_path, mode, encoding="utf-8") as f:
        for item in data:
            # Format as prompt/completion pairs
            formatted_item = {
                "prompt": item['input'],
                "completion": " " + item['output']  # Space at beginning as in example
            }
            f.write(json.dumps(formatted_item, ensure_ascii=False) + "\n")
    print(f"{'Appended to' if append else 'Converted'} {json_path} -> {out_jsonl_path}")

def main():
    # Check if existing data files are present
    existing_data = []
    try:
        if os.path.exists("medical_training_data.json"):
            with open("medical_training_data.json", "r", encoding="utf-8") as fin:
                existing_data = json.load(fin)
            print(f"📂 Found existing data with {len(existing_data)} entries")
    except Exception as e:
        print(f"⚠️ Error reading existing data: {e}")
        existing_data = []
    
    data = []
    failed_urls = []
    successful_count = 0

    print(f"Starting to scrape {len(URLS)} URLs...")
    print("-" * 50)

    for title, url in URLS.items():
        try:
            print(f"Checking: {title}")
            _, overview, symptoms, treatment, medication, when_to_seek = scrape_sections(url)

            # Create a concise treatment summary
            treatment_summary = ""
            
            # Include treatment info if available
            if treatment:
                treatment_lines = treatment.split('\n')
                treatment_summary = treatment_lines[0]  # Get first paragraph
                if len(treatment_summary) > 100:
                    treatment_summary = treatment_summary[:97] + "..."
            
            # Include medication info if available
            med_info = ""
            if medication:
                med_lines = medication.split('\n')
                med_info = med_lines[0]
                if len(med_info) > 80:
                    med_info = med_info[:77] + "..."
            
            # Combine treatment and medication
            if treatment_summary and med_info:
                final_answer = f"{treatment_summary}; {med_info}"
            elif treatment_summary:
                final_answer = treatment_summary
            elif med_info:
                final_answer = med_info
            else:
                # Fallback if no treatment or medication info
                final_answer = "Consult a healthcare provider for proper treatment advice."
                if when_to_seek:
                    final_answer = when_to_seek.split('\n')[0]
                elif symptoms:
                    final_answer = "Treatment depends on symptoms. " + symptoms.split('\n')[0]
                elif overview:
                    final_answer = overview.split('\n')[0]
            
            # Format final answer to be concise and informative
            if len(final_answer) > 200:
                final_answer = final_answer[:197] + "..."
            
            question = f"What is the recommended treatment for {title}?"
            
            entry = {
                "input": question,
                "output": final_answer,
                "type": "treatment",
                "source": url
            }
            data.append(entry)
            successful_count += 1
            print(f"✅ Scraped: {title}")

        except Exception as e:
            failed_urls.append({
                'title': title,
                'url': url,
                'error': str(e)
            })
            print(f"❌ Failed: {title} - {str(e)}")

        time.sleep(2)

    print("-" * 50)
    print(f"Scraping completed!")
    print(f"✅ Successful: {successful_count}")
    print(f"❌ Failed: {len(failed_urls)}")

    if failed_urls:
        print("\n📋 Failed URLs Summary:")
        for item in failed_urls:
            print(f"  - {item['title']}: {item['error']}")

    if data:
        # Combine existing and new data
        combined_data = existing_data + data
        print(f"📊 Total entries: {len(combined_data)} (Added {len(data)} new entries)")
        
        # Write as JSON for training
        with open("medical_training_data.json", "w", encoding="utf-8") as fout:
            json.dump(combined_data, fout, indent=2, ensure_ascii=False)
        print(f"\n💾 Data saved to medical_training_data.json")

        # Convert to JSONL (append if exists)
        if os.path.exists("medical_training_data.jsonl") and existing_data:
            # Create a temp JSON file with only the new data for appending
            with open("temp_new_data.json", "w", encoding="utf-8") as fout:
                json.dump(data, fout, indent=2, ensure_ascii=False)
            convert_json_to_jsonl("temp_new_data.json", "medical_training_data.jsonl", append=True)
            os.remove("temp_new_data.json")
        else:
            # If no JSONL file exists, create it from scratch
            convert_json_to_jsonl("medical_training_data.json", "medical_training_data.jsonl")

        # Handle failed URLs with option to append
        if failed_urls:
            existing_failed = []
            if os.path.exists("failed_urls.json"):
                try:
                    with open("failed_urls.json", "r", encoding="utf-8") as fin:
                        existing_failed = json.load(fin)
                except Exception:
                    existing_failed = []
            
            # Combine existing and new failed URLs
            combined_failed = existing_failed + failed_urls
            
            with open("failed_urls.json", "w", encoding="utf-8") as fout:
                json.dump(combined_failed, fout, indent=2)
            print(f"📋 Failed URLs saved to failed_urls.json")
    else:
        print("⚠️  No new data was successfully scraped!")

if __name__ == "__main__":
    main()