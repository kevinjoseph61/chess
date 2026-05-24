"""
LLM-powered post-game analysis using Groq API.

Analyzes a completed game's PGN and provides personalized,
natural language coaching on missed tactics and strategic concepts.
"""

from django.conf import settings
from groq import Groq


def analyze_game(pgn: str) -> dict:
    """
    Analyze a chess game PGN using Groq's LLM.
    
    Returns a dict with:
        - summary: brief game overview
        - analysis: detailed move-by-move coaching
        - tactics: missed tactical opportunities
        - tips: personalized improvement tips
        - error: error message if analysis failed
    """
    if not settings.GROQ_API_KEY:
        return {"error": "Groq API key not configured"}

    client = Groq(api_key=settings.GROQ_API_KEY)

    prompt = f"""You are an expert chess coach analyzing a student's game. The student played as White.

Analyze this PGN and provide coaching feedback:

{pgn}

Provide your analysis in this exact format:

## Game Summary
A 2-3 sentence overview of the game — opening played, key turning points, and result.

## Critical Moments
Identify the 3-5 most important moments in the game. For each:
- The move number and move played
- What was the best move instead (if applicable)
- Why the best move is stronger (tactical or strategic reasoning)

## Missed Tactics
List any tactical patterns the student missed (forks, pins, skewers, discovered attacks, etc.). If none were missed, say so.

## Strategic Themes
What strategic concepts were relevant in this game? (pawn structure, piece activity, king safety, etc.)

## Top 3 Tips
Three specific, actionable tips for improvement based on this game.

Keep your language encouraging and educational. Use chess notation (e.g., Nf3, e4) when referring to moves."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly, expert chess coach. Analyze games and provide clear, actionable coaching feedback. Be encouraging but honest about mistakes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000,
        )

        analysis_text = completion.choices[0].message.content
        
        # Parse sections
        sections = _parse_sections(analysis_text)
        
        return {
            "raw": analysis_text,
            "summary": sections.get("Game Summary", ""),
            "critical_moments": sections.get("Critical Moments", ""),
            "tactics": sections.get("Missed Tactics", ""),
            "strategy": sections.get("Strategic Themes", ""),
            "tips": sections.get("Top 3 Tips", ""),
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


def _parse_sections(text: str) -> dict:
    """Parse markdown sections from the LLM response."""
    sections = {}
    current_key = None
    current_lines = []
    
    for line in text.split('\n'):
        if line.startswith('## '):
            if current_key:
                sections[current_key] = '\n'.join(current_lines).strip()
            current_key = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    
    if current_key:
        sections[current_key] = '\n'.join(current_lines).strip()
    
    return sections


def coach_move(pgn: str, last_move_category: str, player_color: str = "white") -> dict:
    """
    Provide live coaching feedback for the player's last move.
    Called for notable moves (inaccuracy, mistake, blunder, brilliant).
    
    Returns:
        - tip: short coaching message (1-2 sentences)
        - error: error message if failed
    """
    if not settings.GROQ_API_KEY:
        return {"error": "Groq API key not configured"}

    client = Groq(api_key=settings.GROQ_API_KEY)

    category_context = {
        "brilliant": "The player just made an excellent move.",
        "good": "The player made a solid move.",
        "inaccuracy": "The player made a slight inaccuracy.",
        "mistake": "The player made a mistake that lost some advantage.",
        "blunder": "The player made a serious blunder.",
    }

    context = category_context.get(last_move_category, "")

    prompt = f"""You are a chess coach watching a live game. The student plays {player_color}.

Current game PGN: {pgn}

{context}

Give a brief coaching tip (1-2 sentences max) about the player's LAST move. Be specific about the position — mention piece names, squares, and tactical/strategic ideas. If it was a mistake, briefly mention what would have been better. If it was brilliant, explain why.

Reply with ONLY the coaching tip, no headers or formatting."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise chess coach giving live commentary. Keep responses to 1-2 sentences. Be specific about pieces and squares."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=150,
        )

        tip = completion.choices[0].message.content.strip()
        return {"tip": tip}

    except Exception as e:
        return {"error": str(e)}
