---
title: RLHF Feedback App
emoji: 😻
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.23.2
app_file: app.py
pinned: false
license: mit
short_description: RLHF pipeline
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# RLHF Feedback Application

A Gradio application for collecting human feedback on AI responses for reinforcement learning.

## Features
- Real-time feedback collection
- Firebase integration for data storage
- MT5-based language model
- Gradio interface for easy interaction

## Environment Variables
Required environment variables for Hugging Face Space:
- `FIREBASE_CREDENTIALS`: Firebase service account credentials JSON

## Model
Uses the `ArsenKe/MT5_large_finetuned_chatbot` model for response generation.