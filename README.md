#Medical AI Advisor
A lightweight, locally‑run medical assistant powered by a custom fine‑tuned LLM using Ollama.
Built to provide safe, structured guidance for simple, non‑emergency health concerns using curated doctor–patient dialogue datasets.

# Project Overview
Medical AI Advisor is an end‑to‑end machine learning project demonstrating:

Data scraping and preprocessing

JSONL dataset creation

Custom model fine‑tuning using Ollama

Safety‑aware response generation

A functional Python interface for user interaction

The goal is to explore how domain‑specific AI models can support basic medical understanding while maintaining safety and clarity.

# Features
Symptom‑based guidance for common illnesses

Structured, doctor‑style explanations

Locally hosted model (no cloud or API dependency)

Custom training pipeline

JSONL dataset support

GPU‑accelerated fine‑tuning (RTX 4070)

Basic safety filters to reduce harmful outputs

Architecture
Code
Data Scraper → Dataset Cleaning → JSONL Formatting → Model Fine‑Tuning (Ollama)
        → Safety Filters → User Interaction (main.py)
