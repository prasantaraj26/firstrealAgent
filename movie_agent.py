#!/usr/bin/env python3
"""
Movie Agent: Fetches top 20 movies from 2025 via TMDB, analyzes each
synopsis with an LLM, categorizes by storyline theme, and rates
psychological complexity on a 1-10 scale.

Requirements:
  TMDB_API_KEY       - free key from https://www.themoviedb.org/settings/api
  OPENROUTER_API_KEY - free key from https://openrouter.ai/keys
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import requests

TMDB_BASE_URL = "https://api.themoviedb.org/3"
_genre_cache: dict[int, str] = {}


@dataclass
class Movie:
    id: int
    title: str
    release_date: str
    overview: str
    genres: list[str]
    vote_average: float
    popularity: float


@dataclass
class MovieAnalysis:
    title: str
    release_date: str
    overview: str
    tmdb_genres: list[str]
    vote_average: float
    storyline_category: str
    storyline_description: str
    psychological_rating: int
    psychological_analysis: str
    key_themes: list[str]


def _tmdb_get(path: str, api_key: str, **params) -> dict:
    response = requests.get(
        f"{TMDB_BASE_URL}{path}",
        params={"api_key": api_key, "language": "en-US", **params},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def _load_genre_cache(api_key: str) -> None:
    global _genre_cache
    if _genre_cache:
        return
    data = _tmdb_get("/genre/movie/list", api_key)
    _genre_cache = {g["id"]: g["name"] for g in data.get("genres", [])}


def _genre_names(genre_ids: list[int], api_key: str) -> list[str]:
    _load_genre_cache(api_key)
    return [_genre_cache.get(gid, "Unknown") for gid in genre_ids]


def fetch_movie_details(movie_id: int, api_key: str) -> Optional[dict]:
    try:
        return _tmdb_get(f"/movie/{movie_id}", api_key)
    except Exception:
        return None


def fetch_top_movies(year: int = 2025, count: int = 20) -> list[Movie]:
    api_key = os.environ.get("TMDB_API_KEY", "").strip()
    if not api_key:
        sys.exit("Error: TMDB_API_KEY environment variable is not set.\n"
                 "Get a free key at https://www.themoviedb.org/settings/api")

    movies: list[Movie] = []
    page = 1

    while len(movies) < count:
        data = _tmdb_get(
            "/discover/movie", api_key,
            sort_by="popularity.desc",
            primary_release_year=year,
            with_original_language="en",
            page=page,
        )
        results = data.get("results", [])
        if not results:
            break

        for item in results:
            if len(movies) >= count:
                break
            genres = _genre_names(item.get("genre_ids", []), api_key)
            overview = item.get("overview", "")
            if len(overview) < 80:
                details = fetch_movie_details(item["id"], api_key)
                if details:
                    overview = details.get("overview", overview)
                    genres = [g["name"] for g in details.get("genres", [])] or genres
            movies.append(Movie(
                id=item["id"],
                title=item["title"],
                release_date=item.get("release_date", ""),
                overview=overview,
                genres=genres,
                vote_average=item.get("vote_average", 0.0),
                popularity=item.get("popularity", 0.0),
            ))

        total_pages = data.get("total_pages", 1)
        page += 1
        if page > total_pages:
            break

    return movies[:count]


def _build_movies_block(movies: list[Movie]) -> str:
    lines = []
    for i, m in enumerate(movies, 1):
        year = m.release_date[:4] if m.release_date else "N/A"
        genres = ", ".join(m.genres) if m.genres else "Unknown"
        synopsis = m.overview or "No synopsis available."
        lines.append(
            f"Movie {i}: {m.title} ({year})\n"
            f"TMDB Genres: {genres}\n"
            f"TMDB Rating: {m.vote_average:.1f}/10\n"
            f"Synopsis: {synopsis}\n"
            f"---"
        )
    return "\n".join(lines)


ANALYSIS_PROMPT = """\
You are a film analyst and psychologist. Analyze the following {n} top movies from \
2025 and provide for each:

1. **Storyline Category** - a creative narrative-theme label (not just genre), e.g.:
   "Redemption Arc", "Identity Crisis", "Survival Against Odds", "Political Conspiracy",
   "Found Family", "Revenge Tragedy", "Existential Quest", "Love and Loss",
   "Coming-of-Age", "Systemic Corruption", "Inner Demon", "Cultural Clash",
   "Human vs. Machine" - or invent a better label for the film.

2. **Psychological Complexity Rating** (integer 1-10):
   1-3  = straightforward narrative, clear moral outcomes
   4-6  = moderate depth, some character nuance and thematic layers
   7-9  = high complexity - moral ambiguity, unreliable narrators, deep psychology
   10   = masterful depth - challenges perception of reality, identity, or ethics

3. **Key Themes** - 2-4 psychological or philosophical themes present in the film.

4. **Psychological Analysis** - 2-3 sentences on the film's psychological presentation.

Movies:
{movies_block}

Respond with a valid JSON array of exactly {n} objects in the SAME ORDER as listed. \
Each object must have ONLY these keys:
  "storyline_category"   : string
  "storyline_description": string (one sentence explaining the label)
  "psychological_rating" : integer (1-10)
  "key_themes"           : array of 2-4 strings
  "psychological_analysis": string (2-3 sentences)

Return ONLY the JSON array - no markdown fences, no commentary."""


def analyze_movies_with_claude(movies: list[Movie]) -> list[MovieAnalysis]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("Error: OPENROUTER_API_KEY environment variable is not set.\n"
                 "Get a free key at https://openrouter.ai/keys")
    prompt = ANALYSIS_PROMPT.format(
        n=len(movies),
        movies_block=_build_movies_block(movies),
    )
    print("\nAnalyzing movies via OpenRouter (nvidia/nemotron-3-super-120b-a12b:free)...")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8000,
        },
        timeout=120,
    )
    if not resp.ok:
        sys.exit(f"OpenRouter error {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    raw_text = resp.json()["choices"][0]["message"]["content"].strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    analyses: list[dict] = json.loads(raw_text)
    return [
        MovieAnalysis(
            title=m.title,
            release_date=m.release_date,
            overview=m.overview,
            tmdb_genres=m.genres,
            vote_average=m.vote_average,
            storyline_category=a["storyline_category"],
            storyline_description=a["storyline_description"],
            psychological_rating=int(a["psychological_rating"]),
            psychological_analysis=a["psychological_analysis"],
            key_themes=a["key_themes"],
        )
        for m, a in zip(movies, analyses)
    ]


def display_results(analyzed: list[MovieAnalysis]) -> None:
    ranked = sorted(analyzed, key=lambda m: m.psychological_rating, reverse=True)
    categories: dict[str, list[str]] = {}
    for m in analyzed:
        categories.setdefault(m.storyline_category, []).append(m.title)

    print("\n" + "=" * 80)
    print("TOP 20 MOVIES OF 2025 - STORYLINE CATEGORIES & PSYCHOLOGICAL RATINGS")
    print("=" * 80)
    print("\n  GENRE-CATEGORY OVERVIEW")
    print("-" * 40)
    for cat, titles in sorted(categories.items()):
        print(f"\n  {cat}:")
        for t in titles:
            print(f"    * {t}")

    print("\n\n  MOVIES RANKED BY PSYCHOLOGICAL COMPLEXITY")
    print("=" * 80)
    for rank, m in enumerate(ranked, 1):
        year = m.release_date[:4] if m.release_date else "N/A"
        bar = "#" * m.psychological_rating + "." * (10 - m.psychological_rating)
        synopsis_preview = m.overview[:200] + "..." if len(m.overview) > 200 else m.overview
        print(f"\n{rank:2}. {m.title} ({year})")
        print(f"    TMDB: {m.vote_average:.1f}/10  |  Psych Score: {m.psychological_rating}/10  [{bar}]")
        print(f"    Storyline: {m.storyline_category}")
        print(f"    -> {m.storyline_description}")
        print(f"    TMDB Genres: {', '.join(m.tmdb_genres)}")
        print(f"    Key Themes: {', '.join(m.key_themes)}")
        print(f"    Psychology: {m.psychological_analysis}")
        print(f"    Synopsis: {synopsis_preview}")

    output_path = "movie_analysis_2025.json"
    export = [
        {
            "rank_by_psychology": rank,
            "title": m.title,
            "release_date": m.release_date,
            "tmdb_genres": m.tmdb_genres,
            "tmdb_rating": m.vote_average,
            "storyline_category": m.storyline_category,
            "storyline_description": m.storyline_description,
            "psychological_rating": m.psychological_rating,
            "key_themes": m.key_themes,
            "psychological_analysis": m.psychological_analysis,
            "overview": m.overview,
        }
        for rank, m in enumerate(ranked, 1)
    ]
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(export, fh, indent=2, ensure_ascii=False)
    print(f"\n\nFull analysis saved to {output_path}")


def main() -> None:
    print("Movie Agent - Top 20 Films of 2025")
    print("=" * 50)
    print("\nFetching top 20 movies from TMDB...")
    movies = fetch_top_movies(year=2025, count=20)
    if not movies:
        sys.exit("No movies returned. Check your TMDB_API_KEY and network access.")
    print(f"Retrieved {len(movies)} movies:")
    for i, m in enumerate(movies, 1):
        year = m.release_date[:4] if m.release_date else "N/A"
        print(f"   {i:2}. {m.title} ({year})")
    analyzed = analyze_movies_with_claude(movies)
    display_results(analyzed)


if __name__ == "__main__":
    main()
