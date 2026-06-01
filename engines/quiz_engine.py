import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class QuizQuestion:
    question: str
    correct_answer: str
    question_type: str 
    difficulty: str


QUESTION_BANK: dict[str, list[dict]] = {
    "beginner": [
        {
            "question": "How do you say 'Good morning' in Spanish?",
            "correct_answer": "Buenos días",
            "type": "translate",
        },
        {
            "question": "Fill in the blank: 'Yo ___ (to be) María.'",
            "correct_answer": "soy",
            "type": "fill_blank",
        },
        {
            "question": "What does 'gracias' mean?",
            "correct_answer": "thank you",
            "type": "translate",
        },
        {
            "question": "How do you say 'I have a dog' in Spanish?",
            "correct_answer": "Yo tengo un perro",
            "type": "translate",
        },
        {
            "question": "What is the Spanish word for 'water'?",
            "correct_answer": "agua",
            "type": "translate",
        },
    ],
}


class QuizEngine:
    """Class to take Quick based on initial defined difficulty level"""
    def __init__(self):
        self._asked: set[str] = set()
        self._current: Optional[QuizQuestion] = None

    def next_question(self, level: str) -> Optional[QuizQuestion]:
        bank = QUESTION_BANK.get(level, QUESTION_BANK["beginner"])
        available = [q for q in bank if q["question"] not in self._asked]

        if not available:
            self._asked.clear()  # reset once all questions exhausted
            available = bank

        chosen = random.choice(available)
        self._asked.add(chosen["question"])

        self._current = QuizQuestion(
            question=chosen["question"],
            correct_answer=chosen["correct_answer"],
            question_type=chosen.get("type", "translate"),
            difficulty=level,
        )
        return self._current
