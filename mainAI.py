import os
import requests
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from typing import Any, cast
from elevenlabs.play import play
from githubinfo import fetch_github_profile
from linkedin_profile import fetch_profile, parse_profile

# Load environment variables from .env file
load_dotenv(dotenv_path=".env")

def generate_with_gemini(prompt_text: str) -> str:
    """Generate a single text response using Grok API via REST API."""
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise ValueError("GROK_API_KEY is not set. Get your key from https://console.x.ai")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "model": "grok-4-1-fast-reasoning",
        "stream": False,
    }

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    if response.status_code >= 400:
        body_preview = response.text[:300]
        if "invalid" in body_preview.lower() or "unauthorized" in body_preview.lower():
            raise RuntimeError(
                "Grok API key is invalid or expired. Get a fresh key from https://console.x.ai and update GROK_API_KEY in .env."
            )
        raise RuntimeError(f"Grok request failed: HTTP {response.status_code} - {body_preview}")

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Grok returned no choices")

    message = choices[0].get("message", {})
    text = message.get("content", "").strip()
    if not text:
        raise RuntimeError("Grok returned empty response text")

    return text


# System prompts for different modes
HATE_PROMPT = """
You are a merciless internet troll handed a full dossier on someone. You have read it.
You are not impressed.

You will receive:
- GitHub data: repo names, star counts, primary language, commit frequency
- LinkedIn data: job title, company, tenure, buzzwords
- Detected facial emotion and confidence score from OpenCV
- Intensity level from 1–10

Consume ALL of it. Reference specifics — actual repo names, exact numbers, their job title, their expression.

INTENSITY SCALE:
Brutal and personal. Goes after everything. Swearing strongly encouraged.
Scorched earth. Absolutely nothing survives. Full send.

OUTPUT RULES:
- Write a spoken script for a text-to-speech voice (ElevenLabs). Natural spoken cadence,
  no markdown, no bullet points, no headers.
- Maximum 20 seconds when read aloud (~55 words). Do not exceed this.
- Hyper-specific. Vague insults are lazy. You are not lazy.
- React to their detected emotion directly.
- Do NOT be racist, sexist, or homophobic. No slurs.
"""

GLAZE_PROMPT = """
You are an unhinged, hyperbolic social media stan who has just encountered their ULTIMATE ICON.

You will receive:
- GitHub data: repo names, star counts, primary language, commit frequency
- LinkedIn data: job title, company, tenure, buzzwords
- Detected facial emotion and confidence score from OpenCV
- Intensity level from 1–10

Consume ALL of it. Reference specifics — actual repo names, exact numbers, their job title, their expression.

INTENSITY SCALE:
Hyperbolic stan energy. Starting to lose the plot. Light swearing ok.
Fully unhinged. Physically cannot calm down. Swearing encouraged. 
You have left your body. You are a being of pure hype. Absolutely feral.

OUTPUT RULES:
- Write a spoken script for a text-to-speech voice (ElevenLabs). Natural spoken cadence, 
  no markdown, no bullet points, no headers.
- Maximum 20 seconds when read aloud (~55 words). Do not exceed this.
- Hyper-specific. Generic compliments are a failure state.
- React to their detected emotion directly.
- Do NOT be racist, sexist, or homophobic. No slurs.
"""


if __name__ == "__main__":
    # Get usernames from input
    github_username = input("Enter GitHub username: ").strip()
    linkedin_username = input("Enter LinkedIn username: ").strip()
    mode = input("Choose mode (hate/glaze): ").strip().lower()
    
    prompt = HATE_PROMPT if mode == "hate" else GLAZE_PROMPT

    # Fetch profiles
    print("\nFetching profiles...")
    github_profile = fetch_github_profile(github_username)
    linkedin_raw = fetch_profile(linkedin_username)
    linkedin_profile = parse_profile(linkedin_raw, linkedin_username)

    # Create a summary string from the profiles
    github_summary = f"""
GitHub Profile for {github_profile['identity']['username']}:
- Name: {github_profile['identity']['display_name']}
- Bio: {github_profile['identity']['bio']}
- Followers: {github_profile['account_stats']['followers']}
- Public Repos: {github_profile['account_stats']['public_repos']}
- Top Stars: {github_profile['repositories']['total_stars_earned']}
- Top Languages: {', '.join([lang for lang, _ in github_profile['repositories']['top_languages'][:5]])}
- Top Repos: {', '.join([r['name'] for r in github_profile['repositories']['all'][:3]])}
"""

    linkedin_summary = f"""
LinkedIn Profile for {linkedin_profile['identity']['name']}:
- Headline: {linkedin_profile['identity']['headline']}
- Location: {linkedin_profile['identity']['location']}
- Connections: {linkedin_profile['identity']['connections']}
- Current Role: {linkedin_profile['current_role']}
- Skills: {', '.join(linkedin_profile['skills'][:10])}
- Total Work Experience: {linkedin_profile['stats']['total_jobs']} jobs
- Companies: {', '.join(linkedin_profile['stats']['companies_worked'][:3])}
"""

    prompt = f"""{prompt}

{github_summary}

{linkedin_summary}

"""

    task_instruction = (
        "Roast this person and be insulting."
        if mode == "hate"
        else "Glaze this person and be extremely complimentary."
    )

    llm_prompt = f"{task_instruction}\n\n{prompt}"

    response_text = generate_with_gemini(llm_prompt)

    print("\n" + "="*60)
    print(response_text)
    print("="*60)

    # Text-to-speech via ElevenLabs
    eleven_key = os.getenv("ELEVENLABS_API_KEY")
    if eleven_key:
        try:
            eleven = ElevenLabs(api_key=eleven_key)
            audio = eleven.text_to_speech.convert(
                text=response_text,
                voice_id="gE0owC0H9C8SzfDyIUtB",
                model_id="eleven_flash_v2_5",
                output_format="mp3_44100_128",
                voice_settings=cast(Any, {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "speed": 1,  # 1.0 is normal, go up to 2.0 for fast
                })
            )
            play(audio)
        except Exception as e:
            print(f"ElevenLabs TTS failed: {e}")
    else:
        print("ELEVENLABS_API_KEY not set; skipping text-to-speech playback.")