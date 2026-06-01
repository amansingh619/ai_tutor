from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Lesson:
    topic: str
    level: str
    explanation: str
    example_sentences: list[str] = field(default_factory=list)
    vocabulary: list[dict] = field(default_factory=list)


# creting a basic curriculum
#TODO: We can increase the difficulty level later
CURRICULUM: dict[str, list[str]] = {
    "beginner": [
        "Greetings and introductions",
        "Numbers 1-20",
        "Days of the week",
        "Basic verbs: ser, estar, tener",
        "Present tense regular verbs",
        "Colors and adjectives",
        "Family members",
        "Food and drinks",
    ],
}


class LessonEngine:
    """class to teach the defined lanuguage based on progress"""
    def __init__(self, target_language: str, native_language: str):
        self._target = target_language
        self._native = native_language
        self._progress: dict[str, int] = {"beginner": 0, "intermediate": 0, "advanced": 0}

    def next_topic(self, level: str) -> Optional[str]:
        topics = CURRICULUM.get(level, CURRICULUM["beginner"])
        idx = self._progress.get(level, 0)
        if idx >= len(topics):
            return None
        return topics[idx]

    def advance(self, level: str):
        self._progress[level] = self._progress.get(level, 0) + 1

    def build_lesson_prompt(self, topic: str, level: str) -> str:
        return (
            f"Teach the following {self._target} lesson topic to a {level} student "
            f"whose native language is {self._native}. "
            f"Topic: {topic}. "
            "Give a short explanation (2-3 sentences), then 2 example sentences, "
            "then ask the student to try using it."
        )

    def all_topics(self, level: str) -> list[str]:
        return CURRICULUM.get(level, [])
