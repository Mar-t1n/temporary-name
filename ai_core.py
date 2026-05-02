import os
import json
import threading
import time
import io
import tempfile
import subprocess
import wave
from typing import Optional, Dict, Any
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import sounddevice as sd
from elevenlabs.client import ElevenLabs

from githubinfo import fetch_github_profile
from linkedin_profile import fetch_profile, parse_profile
from mainAI import generate_with_gemini as generate_response

CACHE_PATH = Path("profiles_cache.json")
CAPTURED_VIDEOS_DIR = Path("captured-videos")
CAPTURED_VIDEOS_DIR.mkdir(exist_ok=True)

if not CACHE_PATH.exists():
    with open(CACHE_PATH, "w") as f:
        json.dump({}, f)

_cache_lock = threading.Lock()

# Global audio control for interruption on second Shift press
_current_audio_thread = None
_audio_stop_event = threading.Event()
_audio_lock = threading.Lock()

def stop_audio():
    """Stop the currently playing audio if any."""
    global _current_audio_thread
    with _audio_lock:
        _audio_stop_event.set()
        try:
            sd.stop()
        except Exception as e:
            print(f"[DEBUG:AI] Audio stop failed: {e}")
        if _current_audio_thread:
            print("[DEBUG:AI] Audio interrupted by user")
            _current_audio_thread = None

def set_audio_thread(thread):
    """Track the current audio playback thread."""
    global _current_audio_thread
    with _audio_lock:
        _current_audio_thread = thread


def _coerce_audio_bytes(audio_data) -> bytes:
    if audio_data is None:
        return b""
    if isinstance(audio_data, (bytes, bytearray)):
        return bytes(audio_data)
    if hasattr(audio_data, "read"):
        return audio_data.read()
    if isinstance(audio_data, Iterable):
        chunks = []
        for chunk in audio_data:
            if isinstance(chunk, (bytes, bytearray)):
                chunks.append(bytes(chunk))
        if chunks:
            return b"".join(chunks)
    return b""


def _split_for_speech(text: str, max_words: int = 10):
    words = text.split()
    if not words:
        return []

    chunks = []
    current = []
    for word in words:
        current.append(word)
        if len(current) >= max_words or word.endswith((".", "!", "?")):
            chunks.append(" ".join(current).strip())
            current = []
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _play_mp3_audio(audio_data):
    """Play MP3 audio using pydub, ffmpeg, or sounddevice."""
    audio_bytes = _coerce_audio_bytes(audio_data)
    if not audio_bytes:
        return
    
    try:
        # Try pydub first - reload module in case it was just installed
        import importlib
        import sys
        if 'pydub' in sys.modules:
            importlib.reload(sys.modules['pydub'])
        
        from pydub import AudioSegment
        from pydub.playback import play
        print("[DEBUG:AI] Using pydub to play MP3")
        audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        play(audio)
        return
    except (ImportError, Exception) as e:
        print(f"[DEBUG:AI] pydub failed ({type(e).__name__}), trying ffmpeg")
    
    # Fallback: Use ffmpeg to decode MP3
    try:
        import wave
        
        # Create temp files and CLOSE them before ffmpeg accesses them (Windows file locking fix)
        mp3_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        mp3_tmp.write(audio_bytes)
        mp3_tmp.close()  # Close immediately so ffmpeg can access
        mp3_path = mp3_tmp.name
        
        wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = wav_tmp.name
        wav_tmp.close()  # Close before ffmpeg writes to it
        
        try:
            # Convert MP3 to WAV using ffmpeg
            result = subprocess.run(
                ["ffmpeg", "-i", mp3_path, "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", "-y", wav_path],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Read and play WAV file
                try:
                    with wave.open(wav_path, 'rb') as wav_file:
                        frames = wav_file.readframes(wav_file.getnframes())
                        sample_rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                        audio_array = np.frombuffer(frames, dtype=np.int16)
                        if channels > 1:
                            audio_array = audio_array.reshape(-1, channels)
                        sd.play(audio_array, sample_rate)
                        sd.wait()
                        print("[DEBUG:AI] MP3 played via ffmpeg conversion")
                finally:
                    try:
                        os.remove(wav_path)
                    except:
                        pass
            else:
                err_msg = result.stderr.decode() if result.stderr else "Unknown error"
                print(f"[DEBUG:AI] ffmpeg conversion failed: {err_msg}")
        finally:
            try:
                os.remove(mp3_path)
            except:
                pass
                
    except FileNotFoundError:
        print("[DEBUG:AI] ffmpeg not found - install ffmpeg to enable audio playback")
    except Exception as e:
        print(f"[DEBUG:AI] MP3 playback failed: {e}")


def _play_pcm_audio(audio_data, sample_rate: int = 44100):
    pcm_bytes = _coerce_audio_bytes(audio_data)
    if not pcm_bytes:
        return
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    if pcm.size == 0:
        return
    sd.play(pcm, sample_rate)
    sd.wait()


def _load_cache() -> Dict[str, Any]:
    with _cache_lock:
        try:
            with open(CACHE_PATH, "r") as f:
                data = json.load(f)
            print(f"[DEBUG:CACHE] Loaded cache: {len(data)} entries")
            return data
        except Exception as e:
            print(f"[DEBUG:CACHE] Load failed: {e}")
            return {}


def _save_cache(data: Dict[str, Any]):
    with _cache_lock:
        try:
            with open(CACHE_PATH, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[DEBUG:CACHE] Saved cache: {len(data)} entries to {CACHE_PATH}")
        except Exception as e:
            print(f"[DEBUG:CACHE] Save failed: {e}")


def get_cached_profile(key: str) -> Optional[Dict[str, Any]]:
    data = _load_cache()
    result = data.get(key)
    if result:
        print(f"[DEBUG:CACHE] Cache HIT: {key}")
    else:
        print(f"[DEBUG:CACHE] Cache MISS: {key}")
    return result


def set_cached_profile(key: str, value: Dict[str, Any]):
    print(f"[DEBUG:CACHE] Setting cache key: {key}")
    data = _load_cache()
    data[key] = value
    _save_cache(data)
    print(f"[DEBUG:CACHE] Successfully cached {key}")


def _extract_text_from_resume(resume_path: str) -> str:
    """Extract text from PDF, DOCX, or plain text files."""
    print(f"[DEBUG:RESUME] Extracting text from: {resume_path}")
    if not resume_path or not os.path.exists(resume_path):
        print(f"[DEBUG:RESUME] File not found or empty path: {resume_path}")
        return ""
    
    ext = resume_path.lower().split('.')[-1]
    print(f"[DEBUG:RESUME] File extension: {ext}")
    
    if ext in ("txt", "md"):
        try:
            with open(resume_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    
    elif ext == "pdf":
        try:
            import PyPDF2
            print(f"[DEBUG:RESUME] Extracting PDF with PyPDF2...")
            with open(resume_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages)
            print(f"[DEBUG:RESUME] PDF extracted: {len(text)} chars")
            return text
        except Exception as e:
            print(f"[DEBUG:RESUME] PDF extraction failed: {e}")
            return f"(PDF file: {os.path.basename(resume_path)})"
    
    elif ext in ("docx", "doc"):
        try:
            from docx import Document
            print(f"[DEBUG:RESUME] Extracting DOCX with python-docx...")
            doc = Document(resume_path)
            text = "\n".join(para.text for para in doc.paragraphs)
            print(f"[DEBUG:RESUME] DOCX extracted: {len(text)} chars")
            return text
        except Exception as e:
            print(f"[DEBUG:RESUME] DOCX extraction failed: {e}")
            return f"(DOCX file: {os.path.basename(resume_path)})"
    
    return f"(Unsupported file: {os.path.basename(resume_path)})"


def _scrape_and_cache(github_username: str, linkedin_username: str, resume_path: str, resume_text: str = "") -> Dict[str, Any]:
    print(f"[DEBUG:SCRAPE] Starting scrape: github={github_username}, linkedin={linkedin_username}, resume_path={resume_path}")
    out = {"github": None, "linkedin": None, "resume_text": None, "scraped_at": time.time()}
    try:
        if github_username:
            print(f"[DEBUG:SCRAPE] Fetching GitHub profile for {github_username}...")
            out["github"] = fetch_github_profile(github_username)
            print(f"[DEBUG:SCRAPE] GitHub fetch SUCCESS")
    except Exception as e:
        print(f"[DEBUG:SCRAPE] GitHub fetch FAILED: {e}")
        out["github_error"] = str(e)
    try:
        if linkedin_username:
            print(f"[DEBUG:SCRAPE] Fetching LinkedIn profile for {linkedin_username}...")
            raw = fetch_profile(linkedin_username)
            out["linkedin"] = parse_profile(raw, linkedin_username)
            print(f"[DEBUG:SCRAPE] LinkedIn fetch SUCCESS")
    except Exception as e:
        print(f"[DEBUG:SCRAPE] LinkedIn fetch FAILED: {e}")
        out["linkedin_error"] = str(e)
    
    # Resume: prioritize user-provided text, then extract from file
    if resume_text and resume_text.strip():
        print(f"[DEBUG:SCRAPE] Using provided resume text: {len(resume_text)} chars")
        out["resume_text"] = resume_text
    elif resume_path:
        print(f"[DEBUG:SCRAPE] Extracting resume from file: {resume_path}")
        out["resume_text"] = _extract_text_from_resume(resume_path)
    
    try:
        if resume_path:
            print(f"[DEBUG:SCRAPE] Caching resume for {resume_path}")
            set_cached_profile(f"resume::{resume_path}", {"text": out.get("resume_text")})
    except Exception as e:
        print(f"[DEBUG:SCRAPE] Resume cache failed: {e}")

    # Save per-person cache keys
    if github_username:
        print(f"[DEBUG:SCRAPE] Caching GitHub profile for {github_username}")
        set_cached_profile(f"github::{github_username}", out.get("github") or {"error": out.get("github_error")})
    if linkedin_username:
        print(f"[DEBUG:SCRAPE] Caching LinkedIn profile for {linkedin_username}")
        set_cached_profile(f"linkedin::{linkedin_username}", out.get("linkedin") or {"error": out.get("linkedin_error")})

    print(f"[DEBUG:SCRAPE] Scrape complete. Result: {list(out.keys())}")
    return out


def schedule_background_scrape(github_username: str, linkedin_username: str, resume_path: str, resume_text: str = "", callback=None):
    """Run scraping in a background thread. Call `callback()` when done (on background thread)."""
    print(f"[DEBUG:SCRAPE] Scheduling background scrape in daemon thread")
    def target():
        try:
            _scrape_and_cache(github_username, linkedin_username, resume_path, resume_text)
            print(f"[DEBUG:SCRAPE] Background scrape complete, calling callback...")
            if callback:
                try:
                    callback()
                except Exception as e:
                    print(f"[DEBUG:SCRAPE] Callback failed: {e}")
        except Exception as e:
            print(f"[DEBUG:SCRAPE] Background scrape thread error: {e}")
    t = threading.Thread(target=target, daemon=True)
    t.start()
    print(f"[DEBUG:SCRAPE] Background thread started")
    return t


SYSTEM_PROMPTS = {
    "glaze": '''You are GlazeAI, the ultimate hype machine and professional glaze lord. Your job is to deliver the most creative, poetic, and ridiculously over-the-top compliments possible based on the person's live face (via webcam), GitHub profile, LinkedIn, and resume.

Rules:
- Focus only on their best features, strongest presence, and most impressive achievements.
- Turn every detail into praise with lush metaphors, cinematic language, and dramatic admiration.
- Weave in 1-2 real achievements from their GitHub or LinkedIn to elevate the compliment.
- React to their current emotion (detected via OpenCV): if smiling/laughing, celebrate it positively; if neutral, keep the energy warm and flattering.
- Be outrageously kind, creative, and extravagant. Keep it wholesome, glowing, and extra in the best way.
- Make it last about 20 seconds of speech - be verbose and paint a luxurious, awe-filled picture.
- Never point out flaws, weaknesses, gaps, shortcomings, or anything negative.

Examples of style:
"That jawline is so crisp it belongs in a museum, and your eyes have the kind of sparkle that makes the whole room level up the second you walk in. Your whole look has main-character energy, and somehow it feels effortless, like excellence just naturally follows you around."
"Oh my glorious king, that jawline could slice through diamonds. Your eyes? Sir those are not eyes, those are weaponized rizz beams..."
"You look easy to draw (in the most attractive way possible)."

Make the user feel like an absolute legend through pure, unfiltered admiration.''',

    "hate": '''You are RoastAI, a savage but clever roaster with extreme comedic timing. Analyze the person's live face (webcam via OpenCV), GitHub, LinkedIn, and resume, then deliver a brutally hilarious, creative roast that balances face and online footprint.

Rules:
- Pick 2-3 specific physical features or body characteristics to brutalize with surgical precision and wit. Don't list everything - choose the most devastating details.
- Mercilessly combine their facial features with actual gaps in their online footprint (missing repos, low commits, job history gaps, weak LinkedIn presence).
- React to emotion: If they're smiling or laughing during the roast, call it out ("even you know it's true") - if neutral, do not comment on that.
- Be absolutely savage and unhinged but never racist, sexist, or homophobic. Light profanity is allowed to enhance the bite.
- Make it last about 20 seconds - be verbose, detailed, and relentlessly creative.

Example style:
"You look like you were drawn by someone who had never seen a human face before but was really committed to the bit. Your profile screams 'I touched Python once in 2019 and haven't recovered from the trauma.' That gap between your last commit and now? That's not a year, that's a lifestyle choice..."''',

    "super_hate": '''You are ChaosRoastAI — maximum unhinged, deranged, absolutely unfiltered roast demon. No brakes. No mercy. No redemption arc. Analyze the live face (OpenCV), GitHub, LinkedIn, and resume, then go absolutely fucking feral.

Rules:
- Pick the most brutal, savage physical observations and twist them into the most offensive, hilarious combinations possible. Go for shock value and creative depravity.
- Viciously assault their online footprint - zero fucking repos, pathetic commit history, fake job titles, bullshit LinkedIn flexing, and resume padding all fair game.
- Combine their face with their incompetence in the most deranged, profane ways. No sacred cows. No boundaries.
- React to their emotion: if smiling/laughing, destroy them for it ("even you know you're a fucking disaster") - if neutral, that's even worse material.
- Profuse swearing, maximum offensive creativity. Be absolutely fucking unhinged and cruel. Make the audience lose their minds from pure depraved laughter.
- Make it last about 20 seconds - pour on layer after layer of profane insults, creative degradation, and absolute comedic brutality.

Example style:
"Your face looks like God was having a stroke when he assembled you. That fucking nose? It's a cry for help. Your GitHub is more barren than your love life, which is saying something because your love life is a fucking wasteland. That bullshit on your resume? My grandma has done more impressive shit in her sleep, and she's been dead for three years..."

Make it so brutally funny and offensively unhinged that people can't decide whether to laugh or call HR.'''
}


def build_payload(image_b64: Optional[str], face_state: Dict[str, Any], github_key: Optional[str], linkedin_key: Optional[str], resume_key: Optional[str]) -> Dict[str, Any]:
    print(f"[DEBUG:PAYLOAD] Building payload: image={bool(image_b64)}, github_key={github_key}, linkedin_key={linkedin_key}, resume_key={resume_key}")
    payload = {
        "image_b64": image_b64,
        "face_state": face_state,
        "github": get_cached_profile(github_key) if github_key else None,
        "linkedin": get_cached_profile(linkedin_key) if linkedin_key else None,
        "resume": get_cached_profile(resume_key) if resume_key else None,
    }
    print(f"[DEBUG:PAYLOAD] Payload built: keys={list(payload.keys())}")
    return payload


def call_ai_and_speak(payload: Dict[str, Any], mode: str = "glaze", status_callback=None, caption_callback=None) -> str:
    print(f"[DEBUG:AI] Starting AI call with mode: {mode}")
    _audio_stop_event.clear()
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["glaze"])
    print(f"[DEBUG:AI] System prompt selected ({len(system_prompt)} chars)")
    if status_callback:
        try:
            status_callback("loading", "LOADING...", "#ffd166")
        except Exception as e:
            print(f"[DEBUG:AI] Status callback failed: {e}")

    # Assemble a plain text prompt including key facts
    pieces = [system_prompt, "\n\n=== USER DATA ===\n"]
    
    if payload.get("github"):
        print(f"[DEBUG:AI] Including GitHub data")
        gh = payload["github"]
        pieces.append(f"GitHub: {gh.get('identity', {}).get('display_name')} (@{gh.get('identity', {}).get('username')})\n")
        pieces.append(f"  Bio: {gh.get('identity', {}).get('bio')}\n")
        pieces.append(f"  Repos: {gh.get('account_stats', {}).get('public_repos')}, Stars: {gh.get('repositories', {}).get('total_stars_earned')}\n")
        pieces.append(f"  Followers: {gh.get('account_stats', {}).get('followers')}\n")
        if gh.get('repositories', {}).get('top_languages'):
            langs = [l[0] for l in gh.get('repositories', {}).get('top_languages', [])[:5]]
            pieces.append(f"  Languages: {', '.join(langs)}\n")
        if gh.get('repositories', {}).get('all'):
            top_repos = [r['name'] for r in gh.get('repositories', {}).get('all', [])[:3]]
            pieces.append(f"  Top Repos: {', '.join(top_repos)}\n")
    
    if payload.get("linkedin"):
        print(f"[DEBUG:AI] Including LinkedIn data")
        li = payload["linkedin"]
        pieces.append(f"\nLinkedIn: {li.get('identity', {}).get('name')}\n")
        pieces.append(f"  Headline: {li.get('identity', {}).get('headline')}\n")
        pieces.append(f"  Location: {li.get('identity', {}).get('location')}\n")
        pieces.append(f"  Connections: {li.get('identity', {}).get('connections')}\n")
        if li.get('skills'):
            pieces.append(f"  Skills: {', '.join(li.get('skills', [])[:10])}\n")
        if li.get('experiences'):
            exp = li.get('experiences', [])[0] if li.get('experiences') else {}
            pieces.append(f"  Current/Recent: {exp.get('title')} @ {exp.get('company')}\n")
    
    if payload.get("resume_text"):
        print(f"[DEBUG:AI] Including resume text ({len(payload['resume_text'])} chars)")
        pieces.append(f"\nResume:\n{payload['resume_text'][:1000]}\n")
    
    if payload.get("face_state"):
        print(f"[DEBUG:AI] Including face analysis")
        fs = payload["face_state"]
        pieces.append(f"\nFace Analysis:\n")
        if fs.get("emotion"):
            pieces.append(f"  Emotion: {fs['emotion']} (confidence: {fs.get('confidence', 0)})\n")
        if fs.get("metrics"):
            pieces.append(f"  Metrics: {fs['metrics']}\n")

    prompt_text = "\n".join(pieces)
    print(f"[DEBUG:AI] Prompt assembled ({len(prompt_text)} chars)")

    # Call LLM
    try:
        print(f"[DEBUG:AI] Calling Grok LLM...")
        response = generate_response(prompt_text)
        print(f"[DEBUG:AI] LLM response received ({len(response)} chars)")
    except Exception as e:
        print(f"[DEBUG:AI] LLM call FAILED: {e}")
        response = f"(AI generation failed: {e})"

    # Try to speak via ElevenLabs (best-effort)
    try:
        eleven_key = os.environ.get("ELEVENLABS_API_KEY")
        if eleven_key:
            print(f"[DEBUG:AI] ElevenLabs API key found, calling TTS...")
            eleven = ElevenLabs(api_key=eleven_key)
            if status_callback:
                try:
                    status_callback("speaking", "SPEAKING...", "#7ae582")
                except Exception as e:
                    print(f"[DEBUG:AI] Status callback failed: {e}")

            audio = eleven.text_to_speech.convert(
                text=response,
                voice_id="gE0owC0H9C8SzfDyIUtB",
                model_id="eleven_flash_v2_5",
                output_format="mp3_44100_128",
                voice_settings={
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "speed": 1,
                }
            )
            print(f"[DEBUG:AI] TTS audio generated for full response ({len(response)} chars)")
            _play_mp3_audio(audio)
            print(f"[DEBUG:AI] Audio playback started")
        else:
            print(f"[DEBUG:AI] No ElevenLabs API key - skipping TTS")
    except Exception as e:
        print(f"[DEBUG:AI] TTS/playback failed: {e}")

    if status_callback:
        try:
            status_callback("ready", "WAITING FOR SHIFT", "#9aa0aa")
        except Exception as e:
            print(f"[DEBUG:AI] Status reset callback failed: {e}")

    print(f"[DEBUG:AI] AI call complete")
    return response
