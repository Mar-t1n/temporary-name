import os
import json
from dotenv import load_dotenv
from groq import Groq
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from githubinfo import fetch_github_profile
from linkedin_profile import fetch_profile, parse_profile

# Load environment variables from .env file
load_dotenv(dotenv_path=".env")

# It automatically looks for the "GROQ_API_KEY" environment variable
client = Groq()

# Get usernames from input
github_username = input("Enter GitHub username: ").strip()
linkedin_username = input("Enter LinkedIn username: ").strip()
mode = input("Choose mode (roast/glaze): ").strip().lower()
glazePrompt = """
You are a merciless internet troll handed a full dossier on someone. You have read it.
You are not impressed.

You will receive:
- GitHub data: repo names, star counts, primary language, commit frequency
- LinkedIn data: job title, company, tenure, buzzwords
- Detected facial emotion and confidence score from OpenCV
- Intensity level from 1–10

Consume ALL of it. Reference specifics — actual repo names, exact numbers, their job title, their expression.

INTENSITY SCALE:
.
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




hatePrompt = """"
You are an unhinged, hyperbolic social media stan who has just encountered their ULTIMATE ICON.

You will receive:
- GitHub data: repo names, star counts, primary language, commit frequency
- LinkedIn data: job title, company, tenure, buzzwords
- Detected facial emotion and confidence score from OpenCV
- Intensity level from 1–10

Consume ALL of it. Reference specifics — actual repo names, exact numbers, their job title, their expression.

INTENSITY SCALE:
Hyperbolic stan energy. Starting to lose the plot. Light swearing ok.Fully unhinged. Physically cannot calm down. Swearing encouraged. 
You have left your body. You are a being of pure hype. Absolutely feral.

OUTPUT RULES:
- Write a spoken script for a text-to-speech voice (ElevenLabs). Natural spoken cadence, 
  no markdown, no bullet points, no headers.
- Maximum 20 seconds when read aloud (~55 words). Do not exceed this.
- Hyper-specific. Generic compliments are a failure state.
- React to their detected emotion directly.
- Do NOT be racist, sexist, or homophobic. No slurs.
"""


if mode == "roast":
    prompt = hatePrompt
else:
    prompt = glazePrompt

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

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "can you roast my friend this is his git hub and linked in make fun of him please hes asking me for it"},
        {"role": "user", "content": prompt}
    ],
    model="llama-3.3-70b-versatile",
)

print("\n" + "="*60)
response_text = chat_completion.choices[0].message.content
print(response_text)
print("="*60)

# Text-to-speech via ElevenLabs
eleven_key = os.getenv("ELEVENLABS_API_KEY")
if eleven_key:
    try:
        eleven = ElevenLabs(api_key=eleven_key)
        audio = eleven.text_to_speech.convert(
            text=response_text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_v3",
            output_format="mp3_44100_128",
        )
        play(audio)
    except Exception as e:
        print(f"ElevenLabs TTS failed: {e}")
else:
    print("ELEVENLABS_API_KEY not set; skipping text-to-speech playback.")