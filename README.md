<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/2bf47fed-9175-4f13-a6c9-ec5a76c3088c" /># Voice AI Language Tutor

A voice-first AI tutor that helps learners practice a new language through natural conversation.

Instead of clicking through lessons, users can simply speak to the tutor. The system can teach concepts, run quizzes, answer doubts mid-conversation, evaluate responses, and adapt the learning flow based on learner progress.

This project explores how a conversational AI tutor can behave more like a human teacher while remaining responsive enough for real-time voice interactions.

# Problem Statement

Build a working prototype of a voice-based AI language tutor that:

- Allows a learner to choose a target language
- Teaches new concepts through conversation
- Runs quizzes
- Evaluates spoken responses
- Handles doubts at any point
- Tracks learner progress
- Operates entirely through voice

# Current implementation focuses on:

- English → Spanish learning
- Local STT using Faster-Whisper
- Local TTS using Piper
- LLM-powered tutoring workflow
- LiveKit-based voice orchestration

# Features Implemented
## Voice Conversation
- User speaks naturally
- Voice is transcribed using Faster-Whisper
- Tutor responds through Piper TTS

## Teaching Mode
The tutor can:

- Introduce vocabulary
- Explain grammar
- Teach sentence structures
- Guide learners step-by-step

## Quiz Mode

The tutor can generate:

- Vocabulary quizzes
- Translation quizzes
- Fill-in-the-blank questions
- Pronunciation checks

## Doubt Handling

The learner can interrupt anytime.

## Intent Routing

The system automatically classifies user input into:
- Conversation
- Teaching
- Quiz
- Doubt Clarification

without requiring explicit mode switching.

## Multi-language Detection

Current support:
- English
- Spanish

The system detects the language being spoken and attempts to use the correct TTS voice.

# Architecture
<img width="1024" height="1536" alt="c2603c29-7348-48e3-a271-63ee03688a4a" src="https://github.com/user-attachments/assets/928dec96-26cd-4d0b-a7e9-b50d7dfbe59f" />


# Trade-Offs Made

## Why LiveKit Instead of Pipecat?

LiveKit provided:

- Better voice transport
- Easier interruption handling
- Stronger production ecosystem
- Cleaner agent abstractions

Pipecat is excellent for custom pipelines, but LiveKit allowed faster iteration for this prototype.

## Why Faster-Whisper?

Pros:
- Fully local
- No API costs
- Good multilingual support

Cons:
- CPU latency
- Language detection instability on short utterances

## Why Piper?

Pros:
- Offline
- Lightweight
- Free

Cons:
- Voice quality below premium providers
- Dynamic language switching is harder

# Scaling Strategy

If this evolved into a production system:

## Phase 1

- Current

  - Single user
  - Local models
  - No persistence

## Phase 2

Add:

- SQL Lite
- Redis
- Session history
- Analytics

## Phase 3

Add Model Serving Layer
- vLLM
- Qwen
- Llama
- DeepSeek

Running on GPU instances.

## Phase 4

Adding learning intelligence

Track:

- Vocabulary mastery
- Grammar mastery
- Quiz scores
- Pronunciation quality

Generate personalized lessons.
