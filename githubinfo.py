import requests
from datetime import datetime


def fetch_github_profile(username: str, token: str = None) -> dict:
    """
    Fetch comprehensive GitHub profile data for a given username.
    
    Args:
        username: GitHub username
        token: Optional GitHub personal access token (increases rate limit from 60 to 5000 req/hr)
    
    Returns:
        Dictionary containing all profile data
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def get(url):
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_all_pages(url):
        """Fetch all pages of a paginated endpoint."""
        results = []
        page = 1
        while True:
            response = requests.get(f"{url}?per_page=100&page={page}", headers=headers)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            results.extend(data)
            page += 1
        return results

    base = "https://api.github.com"

    # --- Core user profile ---
    print(f"Fetching profile for '{username}'...")
    user = get(f"{base}/users/{username}")

    # --- Repositories ---
    print("Fetching repositories...")
    repos_raw = get_all_pages(f"{base}/users/{username}/repos")

    repos = []
    for r in repos_raw:
        repos.append({
            "name": r["name"],
            "description": r["description"],
            "language": r["language"],
            "stars": r["stargazers_count"],
            "forks": r["forks_count"],
            "watchers": r["watchers_count"],
            "open_issues": r["open_issues_count"],
            "is_fork": r["fork"],
            "is_archived": r["archived"],
            "topics": r["topics"],
            "url": r["html_url"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "pushed_at": r["pushed_at"],
            "license": r["license"]["name"] if r.get("license") else None,
            "size_kb": r["size"],
        })

    # Sort by stars descending
    repos.sort(key=lambda x: x["stars"], reverse=True)

    # --- Starred repos (what they like) ---
    print("Fetching starred repositories...")
    starred_raw = get_all_pages(f"{base}/users/{username}/starred")
    starred = [
        {
            "name": s["full_name"],
            "description": s["description"],
            "language": s["language"],
            "stars": s["stargazers_count"],
        }
        for s in starred_raw[:50]  # cap at 50
    ]

    # --- Followers & Following ---
    print("Fetching followers and following...")
    followers = [f["login"] for f in get_all_pages(f"{base}/users/{username}/followers")]
    following = [f["login"] for f in get_all_pages(f"{base}/users/{username}/following")]

    # --- Organisations ---
    print("Fetching organizations...")
    orgs_raw = get(f"{base}/users/{username}/orgs")
    orgs = [{"name": o["login"], "description": o.get("description")} for o in orgs_raw]

    # --- Events (recent public activity) ---
    print("Fetching recent activity...")
    events_raw = requests.get(
        f"{base}/users/{username}/events/public?per_page=100", headers=headers
    ).json()

    event_summary = {}
    for e in events_raw:
        t = e["type"]
        event_summary[t] = event_summary.get(t, 0) + 1

    recent_events = []
    for e in events_raw[:20]:
        entry = {"type": e["type"], "repo": e["repo"]["name"], "date": e["created_at"]}
        if e["type"] == "PushEvent":
            commits = e["payload"].get("commits", [])
            entry["commit_messages"] = [c["message"] for c in commits]
        elif e["type"] == "IssuesEvent":
            entry["action"] = e["payload"].get("action")
            entry["issue_title"] = e["payload"].get("issue", {}).get("title")
        elif e["type"] == "PullRequestEvent":
            entry["action"] = e["payload"].get("action")
            entry["pr_title"] = e["payload"].get("pull_request", {}).get("title")
        recent_events.append(entry)

    # --- Derived / computed stats ---
    original_repos = [r for r in repos if not r["is_fork"]]
    forked_repos = [r for r in repos if r["is_fork"]]
    archived_repos = [r for r in repos if r["is_archived"]]

    languages = {}
    for r in original_repos:
        if r["language"]:
            languages[r["language"]] = languages.get(r["language"], 0) + 1
    top_languages = sorted(languages.items(), key=lambda x: x[1], reverse=True)

    total_stars = sum(r["stars"] for r in repos)
    total_forks = sum(r["forks"] for r in repos)

    account_created = datetime.strptime(user["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    account_age_days = (datetime.utcnow() - account_created).days

    # --- Assemble final profile ---
    profile = {
        "identity": {
            "username": user["login"],
            "display_name": user.get("name"),
            "bio": user.get("bio"),
            "company": user.get("company"),
            "location": user.get("location"),
            "email": user.get("email"),
            "website": user.get("blog"),
            "twitter": user.get("twitter_username"),
            "hireable": user.get("hireable"),
            "profile_url": user["html_url"],
            "avatar_url": user["avatar_url"],
        },
        "account_stats": {
            "created_at": user["created_at"],
            "account_age_days": account_age_days,
            "updated_at": user["updated_at"],
            "public_repos": user["public_repos"],
            "public_gists": user["public_gists"],
            "followers": user["followers"],
            "following": user["following"],
        },
        "social": {
            "followers": followers,
            "following": following,
            "follower_count": len(followers),
            "following_count": len(following),
            "follow_ratio": round(len(followers) / max(len(following), 1), 2),
            "organizations": orgs,
        },
        "repositories": {
            "all": repos,
            "original": original_repos,
            "forked": forked_repos,
            "archived": archived_repos,
            "total_stars_earned": total_stars,
            "total_forks_earned": total_forks,
            "top_languages": top_languages,
        },
        "starred": starred,
        "activity": {
            "event_type_summary": event_summary,
            "recent_events": recent_events,
        },
    }

    return profile


def print_summary(profile: dict):
    """Print a human-readable summary of the profile."""
    identity = profile["identity"]
    stats = profile["account_stats"]
    social = profile["social"]
    repos = profile["repositories"]

    print("\n" + "=" * 60)
    print(f"  GitHub Profile: {identity['username']}")
    print("=" * 60)

    print(f"\n👤  Name:       {identity['display_name'] or 'N/A'}")
    print(f"📝  Bio:        {identity['bio'] or 'N/A'}")
    print(f"🏢  Company:    {identity['company'] or 'N/A'}")
    print(f"📍  Location:   {identity['location'] or 'N/A'}")
    print(f"🌐  Website:    {identity['website'] or 'N/A'}")
    print(f"🐦  Twitter:    {identity['twitter'] or 'N/A'}")
    print(f"💼  Hireable:   {identity['hireable']}")

    print(f"\n📅  Account age:    {stats['account_age_days']} days")
    print(f"👥  Followers:      {social['follower_count']}")
    print(f"➡️   Following:      {social['following_count']}")
    print(f"📊  Follow ratio:   {social['follow_ratio']}")
    print(f"⭐  Total stars:    {repos['total_stars_earned']}")
    print(f"🍴  Total forks:    {repos['total_forks_earned']}")
    print(f"📦  Public repos:   {stats['public_repos']}")
    print(f"🍴  Forked repos:   {len(repos['forked'])}")
    print(f"🗄️   Archived repos: {len(repos['archived'])}")

    print(f"\n🗣️  Top Languages:")
    for lang, count in repos["top_languages"][:5]:
        print(f"    - {lang}: {count} repos")

    print(f"\n🏆  Top 5 Repos by Stars:")
    for r in repos["all"][:5]:
        desc = r["description"] or "No description"
        print(f"    ⭐ {r['stars']:>5}  {r['name']}")
        print(f"           {desc[:80]}")

    if social["organizations"]:
        print(f"\n🏛️  Organizations:")
        for org in social["organizations"]:
            print(f"    - {org['name']}: {org['description'] or ''}")

    print(f"\n⚡  Recent Activity:")
    for event_type, count in profile["activity"]["event_type_summary"].items():
        print(f"    {event_type}: {count}")

    print()


if __name__ == "__main__":
    username = input("Enter GitHub username: ").strip()
    token = input("Enter GitHub token (optional, press Enter to skip): ").strip() or None

    try:
        profile = fetch_github_profile(username, token)
        print_summary(profile)

        # Optionally dump full JSON
        import json
        output_file = f"{username}_profile.json"
        with open(output_file, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"Full profile saved to: {output_file}")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Error: User '{username}' not found.")
        elif e.response.status_code == 403:
            print("Error: Rate limit hit. Pass a GitHub token as the second argument.")
        else:
            print(f"HTTP Error: {e}")