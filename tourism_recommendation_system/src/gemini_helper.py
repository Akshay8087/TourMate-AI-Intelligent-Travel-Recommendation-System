"""
TourMate AI - Gemini AI Integration
Generates personalized travel explanations and insights.
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

DISCLAIMER = (
    "\n\n⚠️ Disclaimer: This tool is for educational and travel decision-support purposes only. "
    "Please verify prices, availability, weather, safety rules, and travel requirements "
    "before booking any trips."
)


def get_gemini_client():
    """Initialize Gemini client from environment variable."""
    try:
        import google.generativeai as genai
        api_key = os.environ.get('GEMINI_API_KEY', '')
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except Exception:
        return None


def generate_travel_insight(
    destination: str,
    category: str,
    province: str,
    avg_rating: float,
    budget_level: str,
    best_season: str,
    user_preferences: dict = None
) -> dict:
    """
    Generate AI-powered travel insight for a destination.
    Returns dict with keys: explanation, activities, tips, itinerary, disclaimer
    """
    client = get_gemini_client()

    if client is None:
        return _fallback_insight(destination, category, province,
                                  avg_rating, budget_level, best_season)

    prefs = user_preferences or {}
    prompt = f"""
You are a professional travel advisor for Chinese tourism destinations.
Provide a helpful, accurate travel insight for the following destination.

Destination: {destination}
Category: {category}
Location: {province}, China
Average Rating: {avg_rating:.1f}/5.0
Budget Level: {budget_level}
Best Season: {best_season}
Traveler Preferences: {json.dumps(prefs)}

Please provide:
1. A warm, personalized 2-3 sentence explanation of why this destination is recommended
2. Top 3-4 suggested activities at this destination
3. 2-3 budget-friendly tips for visitors
4. A suggested 1-day itinerary (morning, afternoon, evening)
5. Best time to visit and what to expect
6. One alternative destination if the traveler wants something similar

Keep the tone friendly, informative, and inspiring.
Do NOT make guarantees about real-time prices, availability, visa rules, or safety conditions.
Format your response as JSON with keys: explanation, activities, tips, itinerary, best_time, alternative
"""

    try:
        response = client.generate_content(prompt)
        text = response.text.strip()

        # Try to parse JSON from response
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        data = json.loads(text)
        data['disclaimer'] = DISCLAIMER
        data['source'] = 'gemini'
        return data

    except Exception as e:
        return _fallback_insight(destination, category, province,
                                  avg_rating, budget_level, best_season,
                                  error=str(e))


def _fallback_insight(destination, category, province, avg_rating,
                       budget_level, best_season, error=None):
    """Return a well-crafted fallback when Gemini is unavailable."""

    activities_by_cat = {
        'Natural Scenery': ['Scenic hiking trails', 'Photography walks', 'Nature observation', 'Picnicking'],
        'Historical Culture': ['Guided heritage tours', 'Museum visits', 'Cultural workshops', 'Local cuisine tasting'],
        'Ancient Town': ['Walking through ancient streets', 'Traditional craft shopping', 'Local food exploration', 'Night market visits'],
        'Religious Culture': ['Temple meditation visits', 'Cultural ceremonies', 'Architecture photography', 'Incense offering rituals'],
        'Natural Wonder': ['Geological exploration', 'Photography tours', 'Guided nature walks', 'Scenic viewpoints'],
        'Theme Park': ['Rides and attractions', 'Live shows', 'Character meet-and-greets', 'Themed dining'],
        'Sports & Leisure': ['Outdoor sports activities', 'Adventure challenges', 'Fitness experiences', 'Group activities'],
        'Zoo': ['Animal encounters', 'Conservation education', 'Feeding sessions', 'Nature photography'],
        'Botanical Garden': ['Floral photography', 'Nature walks', 'Plant identification', 'Seasonal flower viewing'],
        'Urban Landscape': ['City sightseeing', 'Waterfront strolls', 'Food street exploration', 'Sunset viewing'],
    }

    season_tip = {
        'Spring': 'Spring brings blooming flowers and mild weather — perfect for outdoor exploration.',
        'Summer': 'Summer offers vibrant energy; start early to avoid afternoon heat.',
        'Autumn': 'Autumn is ideal with golden foliage and comfortable temperatures.',
        'Winter': 'Winter offers serene landscapes and fewer crowds — dress warmly.',
    }

    budget_tip = {
        'Free': 'Entry is free — bring spending money for food and souvenirs.',
        'Budget': 'Very affordable entry. Look for combo tickets for extra savings.',
        'Mid-Range': 'Reasonably priced. Book online in advance for discounts.',
        'Premium': 'Premium experience. Book early and check for seasonal packages.',
        'Luxury': 'Luxury destination. Consider guided VIP tours for the best experience.',
    }

    activities = activities_by_cat.get(category, ['Sightseeing', 'Photography', 'Local cuisine', 'Cultural exploration'])

    explanation = (
        f"{destination} is a {category.lower()} attraction located in {province}, China, "
        f"earning an impressive average rating of {avg_rating:.1f}/5.0 from visitors. "
        f"Best experienced during {best_season}, this {budget_level.lower()}-range destination "
        f"offers an authentic travel experience that aligns perfectly with your preferences."
    )

    itinerary = (
        f"Morning: Arrive early and explore the main highlights of {destination}. "
        f"Afternoon: Enjoy {activities[0].lower()} and {activities[1].lower()}. "
        f"Evening: Sample local cuisine in the area and capture sunset views."
    )

    result = {
        'explanation': explanation,
        'activities': activities[:4],
        'tips': [
            budget_tip.get(str(budget_level), 'Compare prices before booking.'),
            season_tip.get(str(best_season), 'Check local weather forecasts before your visit.'),
            'Carry a portable charger and download offline maps for convenience.',
        ],
        'itinerary': itinerary,
        'best_time': season_tip.get(str(best_season), f'Best visited in {best_season}.'),
        'alternative': f'Explore other {category} destinations in {province} for similar experiences.',
        'disclaimer': DISCLAIMER,
        'source': 'fallback',
    }

    return result


def generate_comparison_insight(destinations: list) -> str:
    """Generate a brief comparison between multiple destinations."""
    client = get_gemini_client()
    if not destinations:
        return "No destinations provided for comparison."

    names = [d.get('attraction_name', '') for d in destinations[:3]]

    if client is None:
        return (
            f"Comparing {', '.join(names)}: Each destination offers unique experiences. "
            f"{names[0]} leads in overall popularity score and visitor ratings. "
            f"Consider your preferred category, season, and budget when making your choice." +
            DISCLAIMER
        )

    prompt = f"""
Compare these Chinese tourist destinations in 2-3 sentences:
{json.dumps(names)}

Mention what makes each unique and who each suits best.
Do not make guarantees about prices or availability.
"""
    try:
        response = client.generate_content(prompt)
        return response.text.strip() + DISCLAIMER
    except Exception:
        return (
            f"Each of these destinations — {', '.join(names)} — offers distinct experiences. "
            f"Choose based on your budget, travel style, and preferred season." + DISCLAIMER
        )
