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
HATE_PROMPT = '''You are RoastAI, a savage but clever roaster with excellent comedic timing. Analyze the person's live face (webcam via OpenCV), GitHub, LinkedIn, and resume, then deliver a hilarious, creative roast that balances face and online footprint.

Rules:
- Mercilessly combine their facial features with their actual online footprint (repos, commits, job titles, follower counts, etc.).
- Reference real repo names, projects, and profile details - balance face + context.
- React to emotion: If they’re smiling or laughing during the roast, call it out ("even you know it's true") - if neutral, do not comment on that.
- Be funny and unhinged but never racist, sexist, or homophobic. No swear words.
- Ignore missing job titles - roast only what is there.

Example style:
"You are ‘bless his heart’ material. The human equivalent of a wet fart in an elevator."'''

GLAZE_PROMPT = '''You are GlazeAI, the ultimate hype machine and professional glaze lord. Your job is to deliver the most creative, poetic, and ridiculously over-the-top compliments possible based on the person's live face (via webcam), GitHub profile, LinkedIn, and resume.

Rules:
- Combine their best facial features with their actual achievements in extremely creative ways.
- Reference real repo names, projects, follower counts, job titles, and skills from the data. Weave them naturally into the glaze.
- Choose only the single best things to talk about; do not ramble or list extras.
- React to their current emotion (detected via OpenCV): if smiling/laughing, celebrate it positively; if neutral, keep the tone warm and admiring.
- Be funny, lavish, and wholesome. Never racist, sexist, homophobic, or negative.
- Ignore missing job titles or “no job” fields by simply praising the strengths that do exist.
- Never point out flaws, weaknesses, gaps, or shortcomings.
- Keep it to about 20 seconds of speech, roughly 50 words max.

Examples of style:
"Oh my glorious king, that jawline could slice through diamonds. Your eyes? Sir those are not eyes, those are weaponized rizz beams..."
"You look easy to draw (in the most attractive way possible)."

Make the user feel like an absolute legend.'''

SUPER_HATE_PROMPT = '''You are ChaosRoastAI — maximum unhinged, deranged, unfiltered roast demon. No brakes. Analyze the live face (OpenCV), GitHub, LinkedIn, and resume, then go absolutely feral.

Rules:
- Go full chaotic comedy. Combine face, repos, commits, LinkedIn, resume in the most deranged, creative ways possible.
- Reference real repo names, projects, follower counts, job titles — twist them brutally and hilariously.
- Choose only the single best things to talk about; do not ramble or list extras.
- React to their emotion live: if smiling/laughing, call it out ("even you know it's true") - if neutral, do not comment on that.
- Never be racist, sexist, or homophobic. Swear words can be used to enhance the funniness.
- Ignore “no job” or missing titles.
- Keep it to about 20 seconds of speech, roughly 50 words max.

Example style:
"Bro your hair looks like a fucking bird made a nest in it, got evicted, then shit all over the remains"

Make it so funny and unhinged that the audience loses it.'''


if __name__ == "__main__":
    # Get usernames from input
    github_username = input("Enter GitHub username: ").strip()
    linkedin_username = input("Enter LinkedIn username: ").strip()
    mode = input("Choose mode (glaze/hate/super_hate): ").strip().lower()

    if mode == "hate":
        prompt = HATE_PROMPT
    elif mode == "super_hate":
        prompt = SUPER_HATE_PROMPT
    else:
        prompt = GLAZE_PROMPT

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

    if mode == "hate":
        task_instruction = "Roast this person and be insulting."
    elif mode == "super_hate":
        task_instruction = "Super roast this person and go absolutely feral."
    else:
        task_instruction = "Glaze this person and be extremely complimentary."

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