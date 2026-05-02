"""
LinkedIn Profile Scraper
Uses Piloterr LinkedIn Profile Info API

Install:
    pip install requests

Usage:
    python linkedin_profile.py
"""

import json
import requests
from datetime import datetime


PILOTERR_API_KEY = "95a25702-ed4e-4fe9-8280-6abf1a0cedb3"

def fetch_profile(username: str) -> dict:
    linkedin_url = f"https://www.linkedin.com/in/{username}/"
    print(f"\nFetching: {linkedin_url}")

    response = requests.get(
        "https://api.piloterr.com/v2/linkedin/profile/info",
        headers={"x-api-key": PILOTERR_API_KEY},
        params={"query": linkedin_url},
    )

    response.raise_for_status()
    return response.json()


def get_field(obj, *keys):
    """Try multiple field name variants and return the first one that has a value."""
    for key in keys:
        val = obj.get(key)
        if val:
            return val
    return None


def parse_profile(data: dict, username: str) -> dict:
    experiences = []
    for exp in data.get("experiences", []):
        experiences.append({
            "title":       get_field(exp, "job_title", "title"),
            "company":     get_field(exp, "company"),
            "from_date":   get_field(exp, "start_date", "start_at"),
            "to_date":     get_field(exp, "end_date", "end_at"),
            "description": get_field(exp, "description"),
            "location":    get_field(exp, "location"),
        })

    educations = []
    for edu in data.get("educations", []):
        educations.append({
            "institution": get_field(edu, "school"),
            "degree":      get_field(edu, "degree", "degree_name"),
            "field":       get_field(edu, "field_of_study"),
            "from_date":   get_field(edu, "start_date", "start_at"),
            "to_date":     get_field(edu, "end_date", "end_at"),
            "description": get_field(edu, "description"),
        })

    projects = []
    for proj in data.get("section_projects", []):
        projects.append({
            "title":       get_field(proj, "title"),
            "from_date":   get_field(proj, "start_at", "start_date"),
            "to_date":     get_field(proj, "end_at", "end_date"),
            "description": get_field(proj, "description"),
            "url":         get_field(proj, "url"),
        })

    certifications = []
    for cert in data.get("section_certifications", []):
        certifications.append({
            "authority": get_field(cert, "authority", "name"),
            "from_date": get_field(cert, "start_at", "start_date"),
            "to_date":   get_field(cert, "end_at", "end_date"),
        })

    skills = []
    raw_skills = data.get("skills", [])
    for s in raw_skills:
        if isinstance(s, dict):
            skills.append(s.get("name"))
        elif isinstance(s, str):
            skills.append(s)

    # Location from address or top-level
    address = data.get("address") or {}
    location_parts = [address.get("city"), address.get("state"), address.get("country")]
    location = ", ".join(p for p in location_parts if p) or data.get("location")

    current_position = data.get("card_current_position") or {}
    current_education = data.get("card_current_education") or {}
    companies = list({e["company"] for e in experiences if e["company"]})
    schools   = list({e["institution"] for e in educations if e["institution"]})

    return {
        "identity": {
            "name":        data.get("full_name"),
            "headline":    data.get("headline"),
            "location":    location,
            "about":       data.get("summary"),
            "connections": data.get("connection_count"),
            "followers":   data.get("follower_count"),
            "website":     data.get("website_url"),
            "profile_url": data.get("profile_url"),
        },
        "current_role":      current_position.get("name"),
        "current_education": current_education.get("name"),
        "experiences":    experiences,
        "educations":     educations,
        "projects":       projects,
        "certifications": certifications,
        "skills":         skills,
        "stats": {
            "total_jobs":       len(experiences),
            "total_schools":    len(educations),
            "total_projects":   len(projects),
            "companies_worked": companies,
            "schools_attended": schools,
        },
        "scraped_at": datetime.utcnow().isoformat(),
    }


def print_summary(profile: dict):
    identity = profile["identity"]
    stats    = profile["stats"]

    print("\n" + "=" * 60)
    print(f"  LinkedIn Profile: {identity['name']}")
    print("=" * 60)

    print(f"\n👤  Name:        {identity['name'] or 'N/A'}")
    print(f"💼  Headline:    {identity['headline'] or 'N/A'}")
    print(f"📍  Location:    {identity['location'] or 'N/A'}")
    print(f"🔗  Connections: {identity['connections'] or 'N/A'}")
    print(f"👥  Followers:   {identity['followers'] or 'N/A'}")
    print(f"🌐  Website:     {identity['website'] or 'N/A'}")
    print(f"\n📝  About:\n    {identity['about'] or 'N/A'}")

    print(f"\n🏢  Current Company: {profile['current_role'] or 'N/A'}")
    print(f"🎓  Current School:  {profile['current_education'] or 'N/A'}")

    print(f"\n💼  Work History ({stats['total_jobs']} jobs):")
    for exp in profile["experiences"]:
        date_range = f"{exp['from_date'] or '?'} → {exp['to_date'] or 'Present'}"
        print(f"    • {exp['title'] or 'Unknown Role'} @ {exp['company'] or 'Unknown'}  [{date_range}]")
        if exp["description"]:
            print(f"      {exp['description'][:120]}...")

    print(f"\n🎓  Education ({stats['total_schools']} schools):")
    for edu in profile["educations"]:
        degree_info = " ".join(filter(None, [edu.get("degree"), edu.get("field")]))
        date_range  = f"{edu['from_date'] or '?'} → {edu['to_date'] or 'Present'}"
        print(f"    • {edu['institution'] or 'Unknown'}  [{date_range}]")
        if degree_info:
            print(f"      {degree_info}")

    if profile["projects"]:
        print(f"\n🚀  Projects ({stats['total_projects']}):")
        for proj in profile["projects"]:
            print(f"    • {proj['title']}")
            if proj["description"]:
                print(f"      {proj['description'][:120]}...")

    if profile["skills"]:
        print(f"\n🛠️  Skills:")
        print(f"    {', '.join(s for s in profile['skills'] if s)}")

    if profile["certifications"]:
        print(f"\n📜  Certifications:")
        for cert in profile["certifications"]:
            print(f"    • {cert['authority']}")

    print()


if __name__ == "__main__":
    username = input("Enter LinkedIn username (e.g. williamhgates): ").strip()

    if "linkedin.com/in/" in username:
        username = username.rstrip("/").split("/in/")[-1]

    try:
        raw     = fetch_profile(username)
        profile = parse_profile(raw, username)
        print_summary(profile)

        output_file = f"{username}_linkedin.json"
        with open(output_file, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"Full profile saved to: {output_file}")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("Error: Invalid API key.")
        elif e.response.status_code == 429:
            print("Error: Free credits used up.")
        else:
            print(f"HTTP Error: {e}")
    except Exception as e:
        print(f"Error: {e}")