import re
from enum import Enum
from langdetect import detect


class Intent(str, Enum):
    TEACHING = "teaching"
    QUIZ = "quiz"
    CONVERSATION = "conversation"
    DOUBT = "doubt"


# Keyword patterns per intent (quick pass before LLM fallback)
_PATTERNS: dict[Intent, list[str]] = {
    Intent.TEACHING: [
        r"\bteach\b", r"\bexplain\b", r"\bhow (do|does|to)\b",
        r"\bwhat (is|are|does)\b", r"\bgrammar\b", r"\brule\b",
        r"\bmeaning\b", r"\bvocabu", r"\btranslate\b",
    ],
    Intent.QUIZ: [
        r"\bquiz\b", r"\btest\b", r"\bexercise\b", r"\bpractice\b",
        r"\bchallenge\b", r"\bask me\b", r"\bcheck\b",
    ],
    Intent.DOUBT: [
        r"\bwhy\b", r"\bi don'?t understand\b", r"\bconfused\b",
        r"\bclarify\b", r"\bdoubt\b", r"\bnot sure\b",
    ],
    Intent.CONVERSATION: [
        r"\btalk\b", r"\bchat\b", r"\blet'?s speak\b",
        r"\bconversation\b", r"\bpractice speaking\b",
    ],
}

# Characters/patterns that strongly suggest a language
_SPANISH_WORDS = re.compile(
    r"\b(hola|gracias|por favor|buenos|buenas|cómo|qué|es|soy|tengo|quiero|hablar)\b",
    re.IGNORECASE,
)


class IntentRouter:
    """
    Class to help in keyword-based intent detection with language identification.
    """

    def detect_language(self, text: str) -> str:
        """
        Returning the language code based on input
        """
        try:
            return detect(text)
        except Exception:
            pass

        if _SPANISH_WORDS.search(text): # fallbacking back
            return "es"
        return "en"

    def route(self, text: str) -> tuple[Intent, str]:
        """
        Returns (intent, language_code) for the given user input.
        """
        lower = text.lower()
        scores: dict[Intent, int] = {i: 0 for i in Intent}

        for intent, patterns in _PATTERNS.items():
            for pat in patterns:
                if re.search(pat, lower):
                    scores[intent] += 1

        best = max(scores, key=lambda i: scores[i])
        intent = Intent.CONVERSATION if scores[best] == 0 else best
        language = self.detect_language(text)
        return intent, language

    def system_prompt_for(
        self,
        intent: Intent,
        target_language: str,
        native_language: str,
        level: str = "beginner",
    ) -> str:
        """Function to return the sytem prompt """
        base = (
            f"You are an expert {target_language} language tutor. "
            f"The student's native language is {native_language}. "
            f"Student level: {level}. "
            "Never use markdown, bullet points, or formatting. Speak naturally. "
        )

        extras = {
            Intent.TEACHING: (
                "Current mode: TEACHING. "
                "Explain clearly with examples. Teach one concept at a time. "
                "Use simple sentences the student can repeat."
                "Always wrap Spanish words or phrases in single quotes, e.g. 'Hola'."
            ),
            Intent.QUIZ: (
                "Current mode: QUIZ. "
                "Ask one question at a time. Wait for the answer, then give feedback. "
                "Adjust difficulty based on responses."
                "Always wrap Spanish words or phrases in single quotes, e.g. 'Hola'. "
            ),
            Intent.CONVERSATION: (
                "Current mode: CONVERSATION. "
                f"Have a natural conversation in {target_language}. "
                "Gently correct mistakes inline. Keep it encouraging."
                "Always wrap Spanish words or phrases in single quotes, e.g. 'Hola'. "
            ),
            Intent.DOUBT: (
                "Current mode: DOUBT RESOLUTION. "
                "The student has a specific question. Answer it directly and clearly. "
                "Give one example to reinforce the explanation."
                "Always wrap Spanish words or phrases in single quotes, e.g. 'Hola'. "
            ),
        }

        return base + extras[intent]
