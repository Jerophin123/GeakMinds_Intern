import pandas as pd
import random
import re
import joblib
import os
import json
import threading
import sys
import time
import webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Import pywebview for embedded browser
try:
    import webview
    WEBVIEW_AVAILABLE = True
except ImportError:
    WEBVIEW_AVAILABLE = False
    print("ERROR: pywebview is not installed!")
    print("Please install it with: pip install pywebview")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


# ============================================================
# SECTION 1: TEXT CLEANING FUNCTION
# ============================================================
# This function cleans and normalizes input text by converting
# to lowercase and removing special characters, keeping only
# letters and whitespace.

def clean_text(text):
    """Clean and normalize text input"""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()


# ============================================================
# SECTION 2: DATASET GENERATION
# ============================================================
# Generates synthetic training data for the ML model with
# examples for each task type: summarization, email, paraphrasing, title.

def generate_dataset(samples_per_class=250):
    """Generate synthetic training dataset"""
    if not isinstance(samples_per_class, int) or samples_per_class < 1:
        samples_per_class = 250

    # Task-specific prompt templates
    summarization = [
        "summarize this paragraph",
        "provide a brief summary",
        "condense this text",
        "give a short overview",
        "summarize this content",
        "provide short summary"
    ]

    email = [
        "rewrite this as professional email",
        "convert this into formal email",
        "draft a business email",
        "make this message professional",
        "turn this into official email",
        "rewrite politely"
    ]

    paraphrasing = [
        "paraphrase this sentence",
        "reword this statement",
        "rewrite without changing meaning",
        "express differently",
        "rephrase clearly",
        "rewrite this sentence"
    ]

    title = [
        "generate a title",
        "create a headline",
        "suggest a blog title",
        "provide article title",
        "generate headline",
        "create clear title"
    ]

    # Generate data samples
    data = []
    for _ in range(samples_per_class):
        data.append([random.choice(summarization), "summarization"])
        data.append([random.choice(email), "email"])
        data.append([random.choice(paraphrasing), "paraphrasing"])
        data.append([random.choice(title), "title"])

    df = pd.DataFrame(data, columns=["text", "label"])
    return df


# ============================================================
# SECTION 3: MODEL TRAINING
# ============================================================
# Trains a Random Forest classifier using TF-IDF vectorization
# to classify text into different task types.

def train_model(df):
    """Train Random Forest classifier model"""
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("DataFrame must contain 'text' and 'label' columns")

    df = df.copy()
    df["text"] = df["text"].apply(clean_text)

    # Remove empty texts after cleaning
    df = df[df["text"].str.len() > 0]
    
    if df.empty:
        raise ValueError("No valid text data after cleaning")

    X = df["text"]
    y = df["label"]

    if len(X) < 2:
        raise ValueError("Insufficient data for train-test split")

    # Split data with stratification, fallback if stratification fails
    try:
        X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    except ValueError:
        # Fallback if stratification fails
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
        random_state=42
    )

    # Create pipeline with TF-IDF vectorizer and Random Forest
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        ))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = pipeline.predict(X_test)
    print("\n========== MODEL EVALUATION ==========")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model - try multiple locations
    saved = False
    save_paths = [
        os.path.join(os.getcwd(), "rf_task_classifier.pkl"),  # Current directory
        "rf_task_classifier.pkl",  # Relative path
    ]
    
    # If running as exe, also try to save in executable directory
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        save_paths.insert(0, os.path.join(exe_dir, "rf_task_classifier.pkl"))
    
    for save_path in save_paths:
        try:
            joblib.dump(pipeline, save_path)
            print(f"\nModel saved as {save_path}")
            saved = True
            break
        except Exception as e:
            continue
    
    if not saved:
        print("\nWarning: Could not save model to any location")

    return pipeline


# ============================================================
# SECTION 4: TEXT PROCESSING FUNCTIONS
# ============================================================
# These functions perform the actual text transformations:
# summarization, email formatting, paraphrasing, and title generation.

def extract_content_from_prompt(text, task_type):
    """Extract actual content from natural language prompts - improved extraction"""
    if not text or not isinstance(text, str):
        return text
    
    original_text = text.strip()
    if not original_text:
        return ""
    
    # More aggressive pattern matching with better regex
    patterns = {
        "1": [
            r"(?i)^(?:summarize|summarise|brief\s+summary|condense|provide\s+a\s+summary|give\s+a\s+summary)\s*(?:this\s+)?(?:paragraph|text|content|article|passage)?\s*[:]\s*(.+)$",
            r"(?i)^(?:summarize|summarise|brief\s+summary|condense|provide\s+a\s+summary|give\s+a\s+summary)\s+(?:this\s+)?(?:paragraph|text|content|article|passage)?\s*[:]\s*(.+)$",
            r"(?i)^(?:summarize|summarise|brief\s+summary|condense|provide\s+a\s+summary|give\s+a\s+summary)\s+(.+)$",
        ],
        "2": [
            r"(?i)^(?:rewrite|convert|draft|format|write|make\s+this)\s+.*?(?:as|into|to)\s+.*?(?:professional|formal|business)\s+.*?(?:email|message)\s*[:]\s*(.+)$",
            r"(?i)^(?:rewrite|convert|draft|format|write|make\s+this)\s+(?:as|into|to)\s+(?:professional|formal|business)\s+(?:email|message)\s*[:]\s*(.+)$",
            r"(?i)^(?:rewrite|convert|draft|format|write|make\s+this)\s+(.+)$",
        ],
        "3": [
            r"(?i)^(?:paraphrase|reword|rewrite|rephrase)\s+(?:this\s+)?(?:sentence|text|statement|paragraph)\s*[:]\s*(.+)$",
            r"(?i)^(?:paraphrase|reword|rewrite|rephrase)\s+(?:this\s+)?(?:sentence|text|statement|paragraph)\s+(.+)$",
            r"(?i)^(?:paraphrase|reword|rewrite|rephrase)\s*[:]\s*(.+)$",
            r"(?i)^(?:paraphrase|reword|rewrite|rephrase)\s+(.+)$",
        ],
        "4": [
            r"(?i)^(?:generate|create|suggest|give|make)\s+.*?(?:title|headline)\s+(?:for|of)\s+(?:this\s+)?(?:content|article|text|blog)?\s*[:]\s*(.+)$",
            r"(?i)^(?:generate|create|suggest|give|make)\s+.*?(?:title|headline)\s*[:]\s*(.+)$",
            r"(?i)^(?:generate|create|suggest|give|make)\s+.*?(?:title|headline)\s+(.+)$",
        ]
    }
    
    # Try patterns for the specific task type
    if task_type in patterns:
        for pattern in patterns[task_type]:
            match = re.search(pattern, original_text, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if extracted and len(extracted) > 5:  # Ensure we got meaningful content
                    return extracted
    
    # More aggressive prefix removal if patterns didn't match
    prefix_patterns = [
        r"^(?i)(?:summarize|summarise|brief\s+summary|condense|provide\s+a\s+summary|give\s+a\s+summary)\s*(?:this\s+)?(?:paragraph|text|content|article|passage)?\s*[:]\s*",
        r"^(?i)(?:rewrite|convert|draft|format|write|make\s+this)\s+.*?(?:as|into|to)\s+.*?(?:professional|formal|business)\s+.*?(?:email|message)\s*[:]\s*",
        r"^(?i)(?:paraphrase|reword|rewrite|rephrase)\s*(?:this\s+)?(?:sentence|text|statement|paragraph)?\s*[:]\s*",
        r"^(?i)(?:generate|create|suggest|give|make)\s+.*?(?:title|headline)\s*(?:for|of)?\s*(?:this\s+)?(?:content|article|text|blog)?\s*[:]\s*",
    ]
    
    cleaned_text = original_text
    for pattern in prefix_patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE | re.DOTALL)
        if cleaned_text != original_text:
            break
    
    # Remove any duplicate prefixes that might remain
    cleaned_text = re.sub(r"^(?i)(?:summarize|summarise|brief\s+summary|condense).*?[:]\s*", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"^(?i)(?:rewrite|convert|draft|format).*?[:]\s*", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"^(?i)(?:paraphrase|reword|rephrase).*?[:]\s*", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"^(?i)(?:generate|create|suggest).*?[:]\s*", "", cleaned_text, flags=re.IGNORECASE)
    
    result = cleaned_text.strip()
    
    # If extraction failed or result is too short, return original (let function handle it)
    if not result or len(result) < 5:
        return original_text
    
    return result


def summarize(text):
    """Summarize text by extracting key sentences - production-ready with error handling"""
    try:
        # Input validation
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        
        text = text.strip()
        if not text:
            return ""
        
        # Aggressively remove any prefixes - multiple passes to catch all variations
        original_text = text
        prefixes = [
            r"^(?i)summarize\s+this\s+paragraph\s*[:]\s*",
            r"^(?i)summarize\s+this\s*[:]\s*",
            r"^(?i)summarize\s*[:]\s*",
            r"^(?i)(?:summarize|summarise|brief\s+summary|condense|provide\s+a\s+summary|give\s+a\s+summary)\s*(?:this\s+)?(?:paragraph|text|content|article|passage)?\s*[:]\s*",
        ]
        
        for prefix in prefixes:
            text = re.sub(prefix, "", text, flags=re.IGNORECASE)
            text = text.strip()
        
        # If no change after prefix removal, try finding content after colon
        if text == original_text and ':' in text:
            parts = text.split(':', 1)
            if len(parts) == 2 and len(parts[1].strip()) > 10:
                text = parts[1].strip()
        
        if not text:
            return ""
        
        # Limit input size for performance (100KB max)
        MAX_INPUT_SIZE = 100000
        if len(text) > MAX_INPUT_SIZE:
            text = text[:MAX_INPUT_SIZE] + "..."
        
        # Split into paragraphs (preserve structure)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        
        # Extract sentences from all paragraphs - improved detection
        all_sentences = []
        # Better sentence splitting that handles various cases
        sentence_endings = re.compile(r'([.!?]+(?:\s+|$))')
        
        for para in paragraphs:
            # Split sentences while preserving punctuation
            parts = sentence_endings.split(para)
            current = ""
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    # This is sentence content
                    current = part.strip()
                else:
                    # This is punctuation/whitespace
                    if current:
                        sentence = (current + part).strip()
                        # Filter very short fragments and ensure it's a real sentence
                        if len(sentence) > 15 and len(sentence.split()) >= 3:
                            all_sentences.append(sentence)
                    current = ""
            # Handle last sentence if it doesn't end with punctuation
            if current and len(current) > 15 and len(current.split()) >= 3:
                # Add period if missing
                if not current.rstrip().endswith(('.', '!', '?')):
                    current += "."
                all_sentences.append(current)
        
        # If sentence splitting failed, try simpler approach
        if not all_sentences:
            # Split by periods, exclamation, question marks
            simple_sentences = re.split(r'[.!?]+\s+', text)
            all_sentences = [s.strip() + "." for s in simple_sentences if s.strip() and len(s.strip()) > 15]
        
        # Handle edge cases
        if not all_sentences:
            # Fallback: return condensed version
            words = text.split()
            target_words = max(1, len(words) // 2)
            result = " ".join(words[:target_words])
            if len(words) > target_words:
                result += "..."
            return result
        
        # Calculate total word count for 50% reduction target
        total_words = sum(len(s.split()) for s in all_sentences)
        target_words = max(1, total_words // 2)  # Target ~50% reduction
        
        # INTELLIGENT SUMMARIZATION - Reduce to ~50% while preserving meaning
        # Strategy: Extract key information from each sentence and condense
        
        def condense_sentence(sentence, max_words):
            """Condense a sentence to max_words while preserving key information"""
            words = sentence.split()
            if len(words) <= max_words:
                return sentence
            
            # Extract key parts: subject + verb + key objects
            # Keep first part (usually contains main info) + important ending
            if max_words >= 8:
                # Take first 60% + last 40% of allowed words
                first_part = words[:int(max_words * 0.6)]
                last_part = words[-int(max_words * 0.4):] if len(words) > max_words else []
                # Combine, avoiding duplicates
                if last_part and last_part[0] not in first_part:
                    result = " ".join(first_part) + " " + " ".join(last_part)
                else:
                    result = " ".join(words[:max_words])
            else:
                result = " ".join(words[:max_words])
            
            # Clean up and add punctuation if needed
            result = result.rstrip('.,!?')
            if not result.endswith(('.', '!', '?')):
                result += "."
            return result
        
        # For single sentence: condense to ~50%
        if len(all_sentences) == 1:
            sentence = all_sentences[0]
            words = sentence.split()
            target = max(1, len(words) // 2)
            result = condense_sentence(sentence, target)
            return result
        
        # For multiple sentences: extract and condense key information
        summary_parts = []
        words_used = 0
        
        # Always include condensed first sentence (introduction) - ~40% of its words
        if all_sentences:
            first = all_sentences[0]
            first_words = first.split()
            first_target = max(3, len(first_words) // 2)  # 50% of first sentence
            condensed_first = condense_sentence(first, first_target)
            summary_parts.append(condensed_first)
            words_used += len(condensed_first.split())
        
        # For 2-3 sentences: add condensed middle/end
        if len(all_sentences) == 2:
            remaining_words = target_words - words_used
            if remaining_words > 5:
                second = all_sentences[1]
                second_words = second.split()
                second_target = min(remaining_words, len(second_words) // 2)
                condensed_second = condense_sentence(second, second_target)
                summary_parts.append(condensed_second)
        elif len(all_sentences) == 3:
            remaining_words = target_words - words_used
            if remaining_words > 5:
                # Add condensed version of last sentence
                last = all_sentences[-1]
                last_words = last.split()
                last_target = min(remaining_words, len(last_words) // 2)
                condensed_last = condense_sentence(last, last_target)
                summary_parts.append(condensed_last)
        
        # For 4-10 sentences: add key middle points
        elif len(all_sentences) <= 10:
            remaining_words = target_words - words_used
            # Add 1-2 key sentences from middle, condensed
            if remaining_words > 8:
                mid_idx = len(all_sentences) // 2
                mid = all_sentences[mid_idx]
                mid_words = mid.split()
                mid_target = min(remaining_words // 2, len(mid_words) // 2)
                if mid_target >= 5:
                    condensed_mid = condense_sentence(mid, mid_target)
                    summary_parts.append(condensed_mid)
                    words_used += len(condensed_mid.split())
                    remaining_words = target_words - words_used
            
            # Add last sentence if space allows
            if remaining_words > 8 and len(all_sentences) > 1:
                last = all_sentences[-1]
                last_words = last.split()
                last_target = min(remaining_words, len(last_words) // 2)
                if last_target >= 5:
                    condensed_last = condense_sentence(last, last_target)
                    summary_parts.append(condensed_last)
        
        # For 11+ sentences: extract key points from different sections
        else:
            remaining_words = target_words - words_used
            sentences_to_include = min(3, len(all_sentences) // 4)  # Max 3 additional sentences
            
            # Get sentences from key positions: 1/3, 1/2, 2/3, end
            key_positions = []
            if len(all_sentences) > 3:
                key_positions.append(len(all_sentences) // 3)
            if len(all_sentences) > 2:
                key_positions.append(len(all_sentences) // 2)
            if len(all_sentences) > 4:
                key_positions.append(len(all_sentences) * 2 // 3)
            if len(all_sentences) > 1:
                key_positions.append(len(all_sentences) - 1)
            
            # Remove duplicates and limit
            key_positions = sorted(list(set(key_positions)))[:sentences_to_include]
            
            words_per_sentence = remaining_words // len(key_positions) if key_positions else remaining_words
            
            for pos in key_positions:
                if remaining_words <= 5:
                    break
                sentence = all_sentences[pos]
                sent_words = sentence.split()
                sent_target = min(words_per_sentence, len(sent_words) // 2, remaining_words)
                if sent_target >= 5:
                    condensed = condense_sentence(sentence, sent_target)
                    summary_parts.append(condensed)
                    remaining_words -= len(condensed.split())
        
        # Join all parts
        result = ". ".join(summary_parts)
        
        # Final check: ensure we're at ~50% word count
        result_words = len(result.split())
        if result_words > total_words * 0.6:  # If still > 60%, condense more
            # Further condense each part
            condensed_parts = []
            for part in summary_parts:
                words = part.split()
                target = max(3, len(words) // 2)
                condensed_parts.append(condense_sentence(part, target))
            result = ". ".join(condensed_parts)
        
        # Ensure proper punctuation
        if not result.rstrip().endswith(('.', '!', '?')):
            result = result.rstrip('.,!?') + "."
        
        return result
        
    except Exception as e:
        # Production error handling - return safe fallback
        return text[:200] + ("..." if len(text) > 200 else "") if text else ""


def format_email(text):
    """Format text as professional email - properly formats content into email structure"""
    try:
        # Input validation
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        
        content = text.strip()
        
        # Remove any remaining prefixes - multiple attempts
        original_content = content
        prefixes = [
            r"^(?i)rewrite\s+this\s+as\s+(?:a\s+)?(?:professional|formal|business)\s+(?:email|message)\s*[:]\s*",
            r"^(?i)convert\s+this\s+(?:to|into)\s+(?:a\s+)?(?:professional|formal|business)\s+(?:email|message)\s*[:]\s*",
            r"^(?i)format\s+this\s+as\s+(?:a\s+)?(?:professional|formal|business)\s+(?:email|message)\s*[:]\s*",
            r"^(?i)(?:rewrite|convert|draft|format|write|make\s+this)\s+.*?(?:as|into|to)\s+.*?(?:professional|formal|business)\s+.*?(?:email|message)\s*[:]\s*",
        ]
        
        for prefix in prefixes:
            content = re.sub(prefix, "", content, flags=re.IGNORECASE)
            content = content.strip()
            if content != original_content:
                break
        
        # If still no change, try finding content after colon
        if content == original_content and ':' in content:
            parts = content.split(':', 1)
            if len(parts) == 2 and len(parts[1].strip()) > 10:
                content = parts[1].strip()
        
        # Final check - if content is still empty or too short, use original
        if not content or len(content) < 5:
            content = original_content
        
        if not content or len(content.strip()) < 5:
            return "Dear Sir/Madam,\n\n\n\nThank you,\n[Your Name]"
        
        # Limit input size for performance (50KB max for emails)
        MAX_INPUT_SIZE = 50000
        if len(content) > MAX_INPUT_SIZE:
            content = content[:MAX_INPUT_SIZE] + "\n\n[Content truncated due to length]"
        
        # Split content into sentences for better formatting
        # Clean content first - normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # First, try splitting by sentence endings (period, exclamation, question mark)
        # Pattern: sentence ending followed by space or end of string
        sentence_pattern = re.compile(r'([.!?]+(?:\s+|$))')
        parts = sentence_pattern.split(content)
        all_sentences = []
        current_sentence = ""
        
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # This is sentence content
                current_sentence = part.strip()
            else:
                # This is punctuation/whitespace
                if current_sentence:
                    complete_sentence = (current_sentence + part).strip()
                    # Only add if it's a meaningful sentence (at least 10 chars)
                    if len(complete_sentence) >= 10:
                        all_sentences.append(complete_sentence)
                    current_sentence = ""
        
        # Handle last sentence if it doesn't end with punctuation
        if current_sentence and len(current_sentence) >= 10:
            # Add period if missing
            if not current_sentence.rstrip().endswith(('.', '!', '?')):
                current_sentence = current_sentence.rstrip() + "."
            all_sentences.append(current_sentence)
        
        # If sentence splitting failed or found very few sentences, try simpler approach
        if len(all_sentences) < 2:
            # Split by periods, exclamation, question marks (more aggressive)
            simple_sentences = re.split(r'[.!?]+\s+', content)
            all_sentences = []
            for s in simple_sentences:
                s = s.strip()
                if s and len(s) >= 10:
                    # Add period if missing
                    if not s.rstrip().endswith(('.', '!', '?')):
                        s = s.rstrip() + "."
                    all_sentences.append(s)
        
        # If still no sentences found, split by commas for very long content
        if not all_sentences and len(content) > 100:
            # Split by commas as last resort for very long run-on sentences
            comma_parts = re.split(r',\s+', content)
            # Group comma-separated parts into sentences
            temp_sentence = ""
            for part in comma_parts:
                part = part.strip()
                if part:
                    if len(temp_sentence) + len(part) < 150:  # Reasonable sentence length
                        temp_sentence += (", " if temp_sentence else "") + part
                    else:
                        if temp_sentence:
                            if not temp_sentence.rstrip().endswith(('.', '!', '?')):
                                temp_sentence = temp_sentence.rstrip() + "."
                            all_sentences.append(temp_sentence)
                        temp_sentence = part
            if temp_sentence:
                if not temp_sentence.rstrip().endswith(('.', '!', '?')):
                    temp_sentence = temp_sentence.rstrip() + "."
                all_sentences.append(temp_sentence)
        
        # Final fallback: use content as single sentence
        if not all_sentences:
            all_sentences = [content]
        
        # Format into professional email paragraphs
        # Capitalize first letter of first sentence
        if all_sentences and len(all_sentences) > 0 and all_sentences[0]:
            first_sentence = all_sentences[0].strip()
            if len(first_sentence) > 0:
                first_char = first_sentence[0]
                if first_char.islower():
                    first_sentence = first_char.upper() + first_sentence[1:]
                    all_sentences[0] = first_sentence
        
        # Clean up all sentences - remove extra whitespace
        cleaned_sentences = []
        for sent in all_sentences:
            if sent:
                # Clean multiple spaces, tabs, newlines
                cleaned = re.sub(r'\s+', ' ', sent).strip()
                if cleaned:
                    cleaned_sentences.append(cleaned)
        
        if not cleaned_sentences:
            cleaned_sentences = [content]
        
        # Group sentences into logical paragraphs (3-4 sentences per paragraph)
        formatted_paragraphs = []
        sentences_per_paragraph = 3
        
        # Process sentences in groups
        for i in range(0, len(cleaned_sentences), sentences_per_paragraph):
            paragraph_sentences = cleaned_sentences[i:i + sentences_per_paragraph]
            if paragraph_sentences:
                # Join sentences with a single space
                paragraph = " ".join(paragraph_sentences)
                # Clean up any remaining multiple spaces
                paragraph = re.sub(r'\s+', ' ', paragraph).strip()
                # Ensure proper punctuation at end
                if paragraph and not paragraph.rstrip().endswith(('.', '!', '?')):
                    paragraph = paragraph.rstrip() + "."
                if paragraph:
                    formatted_paragraphs.append(paragraph)
        
        # If we only have one paragraph but many sentences, split it into 2 paragraphs
        if len(formatted_paragraphs) == 1 and len(cleaned_sentences) > 4:
            # Split sentences roughly in half
            mid_point = len(cleaned_sentences) // 2
            # Ensure we have at least 2 sentences in each paragraph
            if mid_point >= 2 and (len(cleaned_sentences) - mid_point) >= 2:
                first_half = cleaned_sentences[:mid_point]
                second_half = cleaned_sentences[mid_point:]
                
                # Create first paragraph
                first_para = " ".join(first_half)
                first_para = re.sub(r'\s+', ' ', first_para).strip()
                if not first_para.rstrip().endswith(('.', '!', '?')):
                    first_para = first_para.rstrip() + "."
                
                # Create second paragraph
                second_para = " ".join(second_half)
                second_para = re.sub(r'\s+', ' ', second_para).strip()
                if not second_para.rstrip().endswith(('.', '!', '?')):
                    second_para = second_para.rstrip() + "."
                
                formatted_paragraphs = [first_para, second_para]
        
        # Final cleanup: ensure all paragraphs are properly formatted
        final_paragraphs = []
        for para in formatted_paragraphs:
            if para:
                # Final whitespace cleanup
                para = re.sub(r'\s+', ' ', para).strip()
                # Remove any leading/trailing whitespace
                para = para.strip()
                if para:
                    final_paragraphs.append(para)
        
        # If no paragraphs created, use content as single paragraph
        if not final_paragraphs:
            final_content = re.sub(r'\s+', ' ', content).strip()
            if final_content:
                final_paragraphs = [final_content]
        
        # Join paragraphs with double line breaks (email format)
        formatted_content = "\n\n".join(final_paragraphs)
        
        # Build professional email structure
        email = f"""Dear Sir/Madam,

{formatted_content}

Thank you,
[Your Name]"""
        
        return email
        
    except Exception as e:
        # Production error handling - return basic email template with content
        content_fallback = text[:500] if text and len(text) > 5 else "[Content unavailable]"
        return f"""Dear Sir/Madam,

{content_fallback}

Thank you,
[Your Name]"""


def paraphrase(text):
    """Paraphrase text in real-time - comprehensive paraphrasing with sentence restructuring"""
    try:
        # Input validation
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        
        text = text.strip()
        
        # Remove any remaining prefixes
        text = re.sub(r"^(?i)(?:paraphrase|reword|rewrite|rephrase)\s*(?:this\s+)?(?:sentence|text|statement|paragraph)?\s*[:]\s*", "", text, flags=re.IGNORECASE)
        text = text.strip()
        
        if not text:
            return ""
        
        # Limit input size for performance (100KB max)
        MAX_INPUT_SIZE = 100000
        if len(text) > MAX_INPUT_SIZE:
            text = text[:MAX_INPUT_SIZE] + "\n\n[Content truncated]"
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        
        # Comprehensive word and phrase replacement dictionaries
        # Organized for efficient real-time processing
        
        # Word replacements dictionary - comprehensive synonym mapping
        word_replacements_dict = {
            # Verbs - common actions
            'is': 'may be', 'are': 'could be', 'was': 'had been', 'were': 'could have been',
            'can': 'might', 'will': 'could', 'would': 'might', 'should': 'ought to',
            'have': 'possess', 'has': 'contains', 'had': 'owned', 'got': 'obtained',
            'get': 'obtain', 'give': 'provide', 'take': 'acquire', 'put': 'place',
            'make': 'create', 'do': 'perform', 'go': 'proceed', 'come': 'arrive',
            'see': 'observe', 'look': 'examine', 'watch': 'monitor', 'find': 'discover',
            'know': 'understand', 'think': 'believe', 'feel': 'perceive', 'say': 'state',
            'tell': 'inform', 'ask': 'inquire', 'talk': 'discuss', 'speak': 'communicate',
            'help': 'assist', 'use': 'utilize', 'try': 'attempt', 'start': 'begin',
            'stop': 'cease', 'end': 'conclude', 'finish': 'complete', 'work': 'function',
            'need': 'require', 'want': 'desire', 'like': 'prefer', 'love': 'adore',
            'hate': 'despise', 'show': 'demonstrate', 'tell': 'inform', 'explain': 'clarify',
            'learn': 'acquire', 'teach': 'instruct', 'study': 'examine', 'read': 'peruse',
            'write': 'compose', 'draw': 'sketch', 'paint': 'depict', 'build': 'construct',
            'break': 'fracture', 'fix': 'repair', 'buy': 'purchase', 'sell': 'vend',
            'pay': 'compensate', 'cost': 'price', 'spend': 'expend', 'save': 'preserve',
            'keep': 'maintain', 'leave': 'depart', 'stay': 'remain', 'move': 'relocate',
            'run': 'sprint', 'walk': 'stroll', 'sit': 'rest', 'stand': 'remain upright',
            'sleep': 'rest', 'wake': 'awaken', 'eat': 'consume', 'drink': 'imbibe',
            'cook': 'prepare', 'clean': 'sanitize', 'wash': 'cleanse', 'dry': 'dehydrate',
            
            # Adjectives - descriptions
            'good': 'excellent', 'bad': 'poor', 'nice': 'pleasant', 'great': 'outstanding',
            'big': 'large', 'small': 'compact', 'huge': 'enormous', 'tiny': 'minuscule',
            'long': 'extended', 'short': 'brief', 'tall': 'elevated', 'wide': 'broad',
            'narrow': 'confined', 'thick': 'dense', 'thin': 'slender', 'heavy': 'weighty',
            'light': 'illuminated', 'dark': 'dim', 'bright': 'luminous', 'dull': 'lackluster',
            'new': 'recent', 'old': 'previous', 'young': 'youthful', 'fresh': 'novel',
            'clean': 'pristine', 'dirty': 'soiled', 'hot': 'scorching', 'cold': 'frigid',
            'warm': 'temperate', 'cool': 'chilly', 'fast': 'rapid', 'slow': 'gradual',
            'quick': 'swift', 'easy': 'straightforward', 'hard': 'challenging', 'difficult': 'arduous',
            'simple': 'uncomplicated', 'complex': 'intricate', 'important': 'significant',
            'special': 'unique', 'normal': 'standard', 'strange': 'unusual', 'weird': 'peculiar',
            'beautiful': 'attractive', 'ugly': 'unattractive', 'pretty': 'charming',
            'smart': 'intelligent', 'stupid': 'unintelligent', 'clever': 'ingenious',
            'funny': 'humorous', 'sad': 'melancholy', 'happy': 'joyful', 'angry': 'furious',
            'excited': 'enthusiastic', 'bored': 'uninterested', 'tired': 'exhausted',
            'strong': 'powerful', 'weak': 'feeble', 'healthy': 'robust', 'sick': 'ill',
            'rich': 'wealthy', 'poor': 'impoverished', 'free': 'complimentary',
            'expensive': 'costly', 'cheap': 'inexpensive', 'full': 'complete', 'empty': 'vacant',
            'open': 'accessible', 'closed': 'inaccessible', 'right': 'correct', 'wrong': 'incorrect',
            'true': 'accurate', 'false': 'inaccurate', 'real': 'genuine', 'fake': 'artificial',
            'same': 'identical', 'different': 'distinct', 'similar': 'comparable',
            'better': 'superior', 'worse': 'inferior', 'best': 'optimal', 'worst': 'poorest',
            
            # Adverbs
            'very': 'quite', 'really': 'truly', 'quite': 'rather', 'too': 'excessively',
            'so': 'thus', 'also': 'additionally', 'just': 'merely', 'only': 'solely',
            'even': 'still', 'still': 'yet', 'already': 'previously', 'yet': 'nevertheless',
            'again': 'once more', 'always': 'consistently', 'never': 'not ever',
            'often': 'frequently', 'sometimes': 'occasionally', 'usually': 'typically',
            'rarely': 'seldom', 'soon': 'shortly', 'now': 'currently', 'then': 'subsequently',
            'here': 'in this location', 'there': 'in that location', 'where': 'in which location',
            'how': 'in what manner', 'why': 'for what reason', 'when': 'at what time',
            'well': 'effectively', 'badly': 'poorly', 'quickly': 'rapidly', 'slowly': 'gradually',
            'carefully': 'cautiously', 'easily': 'effortlessly', 'hardly': 'barely',
            'almost': 'nearly', 'quite': 'rather', 'pretty': 'fairly', 'rather': 'somewhat',
            
            # Nouns - common objects/concepts
            'person': 'individual', 'people': 'individuals', 'man': 'male', 'woman': 'female',
            'child': 'youngster', 'kid': 'child', 'baby': 'infant', 'adult': 'grown-up',
            'friend': 'companion', 'enemy': 'adversary', 'family': 'relatives', 'home': 'residence',
            'house': 'dwelling', 'room': 'chamber', 'car': 'vehicle', 'bus': 'transport',
            'train': 'locomotive', 'plane': 'aircraft', 'boat': 'vessel', 'bike': 'bicycle',
            'food': 'nourishment', 'water': 'liquid', 'drink': 'beverage', 'money': 'currency',
            'job': 'occupation', 'work': 'employment', 'school': 'educational institution',
            'student': 'learner', 'teacher': 'educator', 'book': 'volume', 'paper': 'document',
            'phone': 'telephone', 'computer': 'device', 'internet': 'network', 'website': 'site',
            'problem': 'issue', 'solution': 'resolution', 'question': 'inquiry', 'answer': 'response',
            'idea': 'concept', 'plan': 'strategy', 'way': 'method', 'thing': 'item',
            'place': 'location', 'time': 'moment', 'day': 'period', 'night': 'evening',
            'week': 'period of seven days', 'month': 'calendar period', 'year': 'annual period',
            'today': 'this day', 'tomorrow': 'the next day', 'yesterday': 'the previous day',
            
            # Prepositions and connectors
            'about': 'concerning', 'above': 'over', 'below': 'under', 'across': 'through',
            'after': 'subsequent to', 'before': 'prior to', 'during': 'throughout',
            'inside': 'within', 'outside': 'beyond', 'between': 'amidst', 'among': 'amid',
            'through': 'via', 'with': 'accompanied by', 'without': 'lacking', 'against': 'opposed to',
            'toward': 'in the direction of', 'towards': 'in the direction of', 'from': 'originating in',
            'to': 'toward', 'for': 'intended for', 'of': 'belonging to', 'in': 'within',
            'on': 'upon', 'at': 'located at', 'by': 'near', 'near': 'close to', 'far': 'distant',
        }
        
        # Phrase replacements dictionary - common phrases
        phrase_replacements_dict = {
            # Time phrases
            'at this point in time': 'now', 'at the present time': 'currently',
            'in the near future': 'soon', 'in the distant future': 'eventually',
            'at the same time': 'simultaneously', 'from time to time': 'occasionally',
            'all the time': 'constantly', 'once in a while': 'periodically',
            
            # Cause and effect
            'due to the fact that': 'because', 'as a result of': 'because of',
            'owing to the fact that': 'because', 'on account of': 'because of',
            'for the reason that': 'because', 'in view of the fact that': 'since',
            'in light of the fact that': 'considering', 'given that': 'since',
            
            # Purpose phrases
            'in order to': 'to', 'for the purpose of': 'to', 'so as to': 'to',
            'with the aim of': 'to', 'with the intention of': 'to',
            'for the sake of': 'to', 'with a view to': 'to',
            
            # Condition phrases
            'in case of': 'if', 'in the event that': 'if', 'provided that': 'if',
            'on condition that': 'if', 'as long as': 'if', 'assuming that': 'if',
            'supposing that': 'if', 'in the event of': 'if',
            
            # Contrast phrases
            'in spite of': 'despite', 'regardless of': 'despite', 'notwithstanding': 'despite',
            'even though': 'although', 'even if': 'although',
            
            # Addition phrases
            'in addition to': 'besides', 'as well as': 'and', 'along with': 'together with',
            'coupled with': 'combined with', 'together with': 'alongside',
            'not to mention': 'in addition', 'let alone': 'much less',
            
            # Reference phrases
            'with regard to': 'regarding', 'with respect to': 'concerning',
            'in terms of': 'regarding', 'as for': 'concerning', 'as to': 'regarding',
            'in relation to': 'concerning', 'with reference to': 'regarding',
            'in connection with': 'related to', 'pertaining to': 'concerning',
            
            # Comparison phrases
            'in comparison to': 'compared to', 'in contrast to': 'unlike',
            'as opposed to': 'unlike', 'rather than': 'instead of',
            
            # Location phrases
            'in the vicinity of': 'near', 'in the neighborhood of': 'around',
            'in close proximity to': 'near', 'at a distance from': 'far from',
            
            # Manner phrases
            'in a manner': 'in a way', 'by means of': 'through', 'by way of': 'via',
            'in the way of': 'regarding', 'in such a way that': 'so that',
            
            # Frequency phrases
            'from time to time': 'occasionally', 'now and then': 'periodically',
            'once in a while': 'occasionally', 'every now and then': 'periodically',
            'time and again': 'repeatedly', 'over and over': 'repeatedly',
            
            # Quantity phrases
            'a great deal of': 'much', 'a large number of': 'many',
            'a lot of': 'many', 'plenty of': 'many', 'a good deal of': 'much',
            'a large amount of': 'much', 'a great many': 'many',
            
            # Quality phrases
            'of great importance': 'important', 'of significant value': 'valuable',
            'highly significant': 'very important', 'extremely important': 'crucial',
            
            # Opinion phrases
            'in my opinion': 'I believe', 'from my perspective': 'I think',
            'it seems to me': 'I think', 'I am of the opinion that': 'I believe',
            'from my point of view': 'I think',
            
            # Conclusion phrases
            'in conclusion': 'finally', 'to sum up': 'in summary', 'in summary': 'briefly',
            'to conclude': 'finally', 'all in all': 'overall', 'on the whole': 'generally',
            
            # Example phrases
            'for instance': 'for example', 'such as': 'like', 'including': 'such as',
            'to give an example': 'for example', 'to illustrate': 'for example',
            
            # Emphasis phrases
            'it is important to note that': 'note that', 'it should be noted that': 'note that',
            'it is worth noting that': 'note that', 'it is essential that': 'must',
            'it is necessary that': 'must', 'it is crucial that': 'must',
            
            # Beginning phrases
            'first of all': 'first', 'to begin with': 'firstly', 'in the first place': 'initially',
            'at first': 'initially', 'to start with': 'firstly',
            
            # Continuation phrases
            'what is more': 'furthermore', 'moreover': 'additionally', 'furthermore': 'in addition',
            'in addition': 'also', 'additionally': 'also', 'besides': 'also',
            
            # Result phrases
            'as a result': 'therefore', 'consequently': 'therefore', 'thus': 'therefore',
            'hence': 'therefore', 'accordingly': 'therefore', 'for this reason': 'therefore',
            
            # Similarity phrases
            'in the same way': 'similarly', 'likewise': 'similarly', 'in a similar manner': 'similarly',
            'correspondingly': 'similarly',
            
            # Exception phrases
            'with the exception of': 'except', 'apart from': 'except', 'other than': 'except',
            'excluding': 'except', 'save for': 'except',
        }
        
        # Generate additional replacements programmatically for comprehensive coverage
        def generate_additional_replacements(base_dict):
            """Generate additional replacement variations - scalable approach"""
            expanded = base_dict.copy()
            
            # Common verb conjugations and variations
            verb_forms = {
                # Past tense
                'used': 'utilized', 'helped': 'assisted', 'made': 'created',
                'got': 'obtained', 'gave': 'provided', 'showed': 'demonstrated',
                'told': 'informed', 'found': 'discovered', 'tried': 'attempted',
                'started': 'began', 'stopped': 'ceased', 'ended': 'concluded',
                'worked': 'functioned', 'played': 'engaged', 'ran': 'sprinted',
                'walked': 'strolled', 'talked': 'discussed', 'said': 'stated',
                'went': 'proceeded', 'came': 'arrived', 'saw': 'observed',
                'looked': 'examined', 'watched': 'monitored', 'heard': 'perceived',
                'felt': 'sensed', 'thought': 'believed', 'knew': 'understood',
                'learned': 'acquired', 'taught': 'instructed', 'studied': 'examined',
                'read': 'perused', 'wrote': 'composed', 'drew': 'sketched',
                'built': 'constructed', 'broke': 'fractured', 'fixed': 'repaired',
                'bought': 'purchased', 'sold': 'vended', 'paid': 'compensated',
                'spent': 'expended', 'saved': 'preserved', 'kept': 'maintained',
                'left': 'departed', 'stayed': 'remained', 'moved': 'relocated',
                'slept': 'rested', 'woke': 'awakened', 'ate': 'consumed',
                'drank': 'imbibed', 'cooked': 'prepared', 'cleaned': 'sanitized',
                
                # Present participle (-ing)
                'using': 'utilizing', 'helping': 'assisting', 'making': 'creating',
                'getting': 'obtaining', 'giving': 'providing', 'showing': 'demonstrating',
                'telling': 'informing', 'finding': 'discovering', 'trying': 'attempting',
                'starting': 'beginning', 'stopping': 'ceasing', 'ending': 'concluding',
                'working': 'functioning', 'playing': 'engaging', 'running': 'sprinting',
                'walking': 'strolling', 'talking': 'discussing', 'saying': 'stating',
                'going': 'proceeding', 'coming': 'arriving', 'seeing': 'observing',
                'looking': 'examining', 'watching': 'monitoring', 'hearing': 'perceiving',
                'feeling': 'sensing', 'thinking': 'believing', 'knowing': 'understanding',
                'learning': 'acquiring', 'teaching': 'instructing', 'studying': 'examining',
                'reading': 'perusing', 'writing': 'composing', 'drawing': 'sketching',
                'building': 'constructing', 'breaking': 'fracturing', 'fixing': 'repairing',
                'buying': 'purchasing', 'selling': 'vending', 'paying': 'compensating',
                'spending': 'expending', 'saving': 'preserving', 'keeping': 'maintaining',
                'leaving': 'departing', 'staying': 'remaining', 'moving': 'relocating',
                'sleeping': 'resting', 'waking': 'awakening', 'eating': 'consuming',
                'drinking': 'imbibing', 'cooking': 'preparing', 'cleaning': 'sanitizing',
            }
            expanded.update(verb_forms)
            
            # Adjective variations
            adj_variations = {
                # Comparative
                'better': 'superior', 'worse': 'inferior', 'bigger': 'larger',
                'smaller': 'more compact', 'faster': 'more rapid', 'slower': 'more gradual',
                'easier': 'more straightforward', 'harder': 'more challenging',
                'nicer': 'more pleasant', 'greater': 'more outstanding',
                'longer': 'more extended', 'shorter': 'more brief', 'taller': 'more elevated',
                'wider': 'broader', 'narrower': 'more confined', 'thicker': 'denser',
                'thinner': 'more slender', 'heavier': 'more weighty', 'lighter': 'brighter',
                'newer': 'more recent', 'older': 'more previous', 'younger': 'more youthful',
                'fresher': 'more novel', 'cleaner': 'more pristine', 'dirtier': 'more soiled',
                'hotter': 'more scorching', 'colder': 'more frigid', 'warmer': 'more temperate',
                'cooler': 'more chilly', 'quicker': 'more swift', 'simpler': 'more uncomplicated',
                'more complex': 'more intricate', 'more important': 'more significant',
                'more special': 'more unique', 'more normal': 'more standard',
                'more strange': 'more unusual', 'more beautiful': 'more attractive',
                'more ugly': 'more unattractive', 'smarter': 'more intelligent',
                'more clever': 'more ingenious', 'funnier': 'more humorous',
                'sadder': 'more melancholy', 'happier': 'more joyful', 'angrier': 'more furious',
                'more excited': 'more enthusiastic', 'more bored': 'more uninterested',
                'more tired': 'more exhausted', 'stronger': 'more powerful', 'weaker': 'more feeble',
                'healthier': 'more robust', 'richer': 'more wealthy', 'poorer': 'more impoverished',
                
                # Superlative
                'best': 'optimal', 'worst': 'poorest', 'biggest': 'largest',
                'smallest': 'most compact', 'fastest': 'most rapid', 'slowest': 'most gradual',
                'easiest': 'most straightforward', 'hardest': 'most challenging',
                'nicest': 'most pleasant', 'greatest': 'most outstanding',
                'longest': 'most extended', 'shortest': 'most brief', 'tallest': 'most elevated',
                'widest': 'broadest', 'narrowest': 'most confined', 'thickest': 'densest',
                'thinnest': 'most slender', 'heaviest': 'most weighty', 'lightest': 'brightest',
                'newest': 'most recent', 'oldest': 'most previous', 'youngest': 'most youthful',
                'freshest': 'most novel', 'cleanest': 'most pristine', 'dirtiest': 'most soiled',
                'hottest': 'most scorching', 'coldest': 'most frigid', 'warmest': 'most temperate',
                'coolest': 'most chilly', 'quickest': 'most swift', 'simplest': 'most uncomplicated',
                'most complex': 'most intricate', 'most important': 'most significant',
                'most special': 'most unique', 'most normal': 'most standard',
                'most strange': 'most unusual', 'most beautiful': 'most attractive',
                'most ugly': 'most unattractive', 'smartest': 'most intelligent',
                'most clever': 'most ingenious', 'funniest': 'most humorous',
                'saddest': 'most melancholy', 'happiest': 'most joyful', 'angriest': 'most furious',
                'most excited': 'most enthusiastic', 'most bored': 'most uninterested',
                'most tired': 'most exhausted', 'strongest': 'most powerful', 'weakest': 'most feeble',
                'healthiest': 'most robust', 'richest': 'most wealthy', 'poorest': 'most impoverished',
            }
            expanded.update(adj_variations)
            
            # Adverb variations
            adv_variations = {
                'quickly': 'rapidly', 'slowly': 'gradually', 'carefully': 'cautiously',
                'easily': 'effortlessly', 'hardly': 'barely', 'really': 'truly',
                'actually': 'genuinely', 'probably': 'likely', 'possibly': 'perhaps',
                'certainly': 'definitely', 'absolutely': 'completely', 'totally': 'entirely',
                'completely': 'fully', 'entirely': 'wholly', 'partially': 'somewhat',
                'mostly': 'mainly', 'usually': 'typically', 'normally': 'generally',
                'rarely': 'seldom', 'often': 'frequently', 'sometimes': 'occasionally',
                'always': 'consistently', 'never': 'not ever', 'forever': 'eternally',
                'immediately': 'instantly', 'soon': 'shortly', 'later': 'subsequently',
                'earlier': 'previously', 'recently': 'lately', 'currently': 'presently',
                'previously': 'formerly', 'suddenly': 'abruptly', 'gradually': 'slowly',
                'quickly': 'swiftly', 'slowly': 'leisurely', 'carefully': 'meticulously',
            }
            expanded.update(adv_variations)
            
            return expanded
        
        # Expand word replacements
        word_replacements_dict = generate_additional_replacements(word_replacements_dict)
        
        # Expand phrase replacements with variations
        phrase_variations = {
            # Additional time variations
            'right now': 'currently', 'at this moment': 'now', 'at present': 'currently',
            'these days': 'nowadays', 'in this day and age': 'currently',
            
            # Additional cause variations
            'thanks to': 'because of', 'as a consequence of': 'because of',
            'on the grounds that': 'because', 'by reason of': 'because of',
            
            # Additional purpose variations
            'with the goal of': 'to', 'aiming to': 'to', 'seeking to': 'to',
            
            # Additional condition variations
            'should it be that': 'if', 'were it to be that': 'if',
            'in the circumstance that': 'if', 'in the situation that': 'if',
            
            # Additional contrast variations
            'despite the fact that': 'although', 'notwithstanding that': 'although',
            'even when': 'although', 'even while': 'although',
            
            # Additional reference variations
            'in the context of': 'regarding', 'in the framework of': 'regarding',
            'within the scope of': 'regarding', 'in the realm of': 'regarding',
            
            # Additional manner variations
            'in the style of': 'like', 'in the fashion of': 'like',
            'after the manner of': 'like', 'in the mode of': 'like',
        }
        phrase_replacements_dict.update(phrase_variations)
        
        # Convert dictionaries to compiled regex patterns for performance
        word_replacements = [
            (re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE), replacement)
            for word, replacement in word_replacements_dict.items()
        ]
        
        phrase_replacements = [
            (re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE), replacement)
            for phrase, replacement in phrase_replacements_dict.items()
        ]
        
        # Sort phrase replacements by length (longest first) for proper matching
        phrase_replacements.sort(key=lambda x: len(x[0].pattern), reverse=True)
        
        # Helper function to preserve capitalization
        def replace_preserve_case(match, replacement):
            matched = match.group(0)
            if matched and matched[0].isupper():
                return replacement.capitalize()
            return replacement
        
        # Process each paragraph
        paraphrased_paragraphs = []
        for para in paragraphs:
            paraphrased = para
            
            # Step 1: Apply phrase replacements first (longer patterns first)
            for pattern, replacement in phrase_replacements:
                paraphrased = pattern.sub(replacement, paraphrased)
            
            # Step 2: Apply word replacements
            for pattern, replacement in word_replacements:
                paraphrased = pattern.sub(
                    lambda m: replace_preserve_case(m, replacement),
                    paraphrased
                )
            
            # Step 3: Restructure common sentence patterns
            # "It is X that Y" -> "Y is X"
            paraphrased = re.sub(
                r'\bIt is (.+?) that (.+?)([.!?]|$)',
                r'\2 is \1\3',
                paraphrased,
                flags=re.IGNORECASE
            )
            
            # "There is/are X" -> "X exists/exist" (for simple cases)
            paraphrased = re.sub(
                r'\bThere is (a |an )?(.+?)([.!?]|$)',
                r'\2 exists\3',
                paraphrased,
                flags=re.IGNORECASE
            )
            paraphrased = re.sub(
                r'\bThere are (.+?)([.!?]|$)',
                r'\1 exist\2',
                paraphrased,
                flags=re.IGNORECASE
            )
            
            # "I have X" -> "I possess X" or "X is with me"
            paraphrased = re.sub(
                r'\bI have (.+?)([.!?]|$)',
                lambda m: f"I possess {m.group(1)}{m.group(2)}" if random.random() > 0.5 else f"{m.group(1)} is with me{m.group(2)}",
                paraphrased,
                flags=re.IGNORECASE
            )
            
            # Step 4: Change sentence beginnings for variation
            sentence_starters = [
                (r'^(I|We|They|He|She|It)\s+', r'Additionally, \1 '),
                (r'^(This|That|These|Those)\s+', r'Furthermore, \1 '),
            ]
            
            # Only change first sentence of paragraph
            first_sentence_match = re.match(r'^([^.!?]+[.!?])', paraphrased)
            if first_sentence_match:
                first_sent = first_sentence_match.group(1)
                rest = paraphrased[len(first_sent):]
                # Randomly add variation
                if random.random() > 0.7:
                    if not first_sent.startswith(('In other words', 'To put it differently', 'That is to say', 'Additionally', 'Furthermore')):
                        variations = ['In other words, ', 'To put it differently, ', 'That is to say, ']
                        first_sent = random.choice(variations) + first_sent[0].lower() + first_sent[1:] if len(first_sent) > 1 else first_sent
                paraphrased = first_sent + rest
            
            # Step 5: Clean up whitespace
            paraphrased = re.sub(r'\s+', ' ', paraphrased).strip()
            
            # Step 6: Ensure we made changes - if not, add variation
            if paraphrased == para and len(paraphrased) > 20:
                intro_phrases = ["In other words, ", "To put it differently, ", "That is to say, ", "Essentially, "]
                intro = random.choice(intro_phrases)
                if len(paraphrased) > 1:
                    paraphrased = intro + paraphrased[0].lower() + paraphrased[1:]
                else:
                    paraphrased = intro + paraphrased.lower()
            
            paraphrased_paragraphs.append(paraphrased)
        
        result = '\n\n'.join(paraphrased_paragraphs)
        
        # Final cleanup
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
        
    except Exception as e:
        # Production error handling
        return text if text else ""


def generate_title(text):
    """Generate meaningful title from text - analyzes entire content for proper title extraction"""
    try:
        # Input validation
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        
        text = text.strip()
        
        # Remove any remaining prefixes
        text = re.sub(r"^(?i)(?:generate|create|suggest|give|make)\s+.*?(?:title|headline)\s*(?:for|of)?\s*(?:this\s+)?(?:content|article|text|blog)?\s*[:]\s*", "", text, flags=re.IGNORECASE)
        text = text.strip()
        
        if not text:
            return ""
        
        # Limit input size for performance (50KB max for title generation)
        MAX_INPUT_SIZE = 50000
        if len(text) > MAX_INPUT_SIZE:
            text = text[:MAX_INPUT_SIZE]
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Comprehensive stop words to filter out
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'in', 'on', 'at', 'by', 'for', 'with', 'to', 'of', 'from', 'as', 'and', 'or', 'but',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
            'am', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'done', 'doing', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'cannot', 'couldn', 'wouldn', 'shouldn', 'won', 'isn', 'aren', 'wasn', 'weren',
            'hasn', 'haven', 'hadn', 'doesn', 'didn', 'here', 'there', 'where', 'when', 'why', 'how',
            'what', 'which', 'who', 'whom', 'whose', 'some', 'any', 'all', 'each', 'every', 'both',
            'few', 'many', 'most', 'other', 'another', 'such', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'now', 'then', 'more', 'most', 'less', 'least', 'much', 'many',
            'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'before',
            'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'during', 'except', 'inside',
            'into', 'near', 'outside', 'over', 'since', 'through', 'throughout', 'toward', 'towards',
            'under', 'underneath', 'until', 'upon', 'within', 'without',
        }
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            sentences = [text]
        
        # Extract key information from all sentences, not just first
        all_words = []
        word_frequency = {}
        
        # Analyze all sentences for key terms
        for sentence in sentences:
            # Extract words (alphanumeric only)
            words = re.findall(r'\b[a-zA-Z]+\b', sentence)
            for word in words:
                word_lower = word.lower()
                # Skip stop words and very short words
                if word_lower not in stop_words and len(word) > 2:
                    all_words.append(word)
                    word_frequency[word_lower] = word_frequency.get(word_lower, 0) + 1
        
        # If no meaningful words found, use first sentence approach
        if not all_words:
            first_sentence = sentences[0] if sentences else text
            # Remove common starting phrases
            first_sentence = re.sub(r'^(This|That|These|Those|The|A|An|Here|There|I|We|You|They)\s+', '', 
                                   first_sentence, flags=re.IGNORECASE)
            words = re.findall(r'\b[a-zA-Z]+\b', first_sentence)
            all_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
            if not all_words:
                all_words = words[:10]
        
        # Score words by importance: frequency + position (earlier = better)
        word_scores = {}
        word_positions = {}  # Track where words appear
        
        # Track positions
        for idx, sentence in enumerate(sentences):
            words_in_sent = re.findall(r'\b[a-zA-Z]+\b', sentence)
            for word in words_in_sent:
                word_lower = word.lower()
                if word_lower not in stop_words and len(word) > 2:
                    if word_lower not in word_positions:
                        word_positions[word_lower] = idx
        
        # Calculate scores
        max_freq = max(word_frequency.values()) if word_frequency else 1
        max_pos = max(word_positions.values()) if word_positions else len(sentences)
        
        for word_lower, freq in word_frequency.items():
            position_score = 1.0 - (word_positions.get(word_lower, max_pos) / max(max_pos, 1))
            frequency_score = freq / max_freq
            # Combine: frequency (70%) + position (30%) - earlier words are more important
            word_scores[word_lower] = (frequency_score * 0.7) + (position_score * 0.3)
        
        # Get top 2-3 words by score
        if word_scores:
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            top_words = []
            seen = set()
            
            for word_lower, score in sorted_words[:5]:  # Get top 5 candidates
                # Find original case version
                for orig_word in all_words:
                    if orig_word.lower() == word_lower and word_lower not in seen:
                        top_words.append(orig_word)
                        seen.add(word_lower)
                        break
                if len(top_words) >= 3:  # We only need 2-3 words
                    break
            
            # Select exactly 2-3 most important words
            if len(top_words) >= 2:
                title_words = top_words[:3]  # Max 3 words, prefer 2-3
            elif len(top_words) == 1:
                # If only one word, try to get one more from frequency
                title_words = top_words
                for word_lower, freq in sorted(word_frequency.items(), key=lambda x: x[1], reverse=True):
                    if word_lower not in seen:
                        for orig_word in all_words:
                            if orig_word.lower() == word_lower:
                                title_words.append(orig_word)
                                break
                        if len(title_words) > len(top_words):
                            break
                if len(title_words) < 2:
                    title_words = top_words
            else:
                title_words = top_words
        else:
            # Fallback: use first 2-3 meaningful words
            seen = set()
            unique_words = []
            for word in all_words:
                word_lower = word.lower()
                if word_lower not in seen:
                    unique_words.append(word)
                    seen.add(word_lower)
                    if len(unique_words) >= 3:
                        break
            title_words = unique_words[:3]
        
        # Ensure we have 2-3 words
        if len(title_words) < 2:
            # Try to get more words from text
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text)  # Words with 4+ letters
            for word in words:
                word_lower = word.lower()
                if word_lower not in stop_words and word not in title_words:
                    title_words.append(word)
                    if len(title_words) >= 3:
                        break
        
        # Final selection: exactly 2-3 words (prefer 2, allow 3)
        if len(title_words) > 3:
            title_words = title_words[:3]
        elif len(title_words) < 2:
            # Last resort: get any meaningful words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            for word in words:
                word_lower = word.lower()
                if word_lower not in stop_words and word not in title_words:
                    title_words.append(word)
                    if len(title_words) >= 2:
                        break
        
        # Build title - exactly 2-3 words
        if not title_words:
            # Ultimate fallback
            words = text.split()
            title_words = [w for w in words if len(w) > 2][:3]
        
        # Limit to 2-3 words
        title_words = title_words[:3]
        
        title = " ".join(title_words)
        
        # Clean up: remove special chars
        title = re.sub(r'[^\w\s\'-]', '', title)
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Title case - capitalize each word
        words_list = title.split()
        if words_list:
            title = " ".join([w.capitalize() for w in words_list])
        
        # Final validation - ensure 2-3 words
        words_list = title.split()
        if len(words_list) > 3:
            title = " ".join(words_list[:3])
        elif len(words_list) < 2 and text:
            # Try one more time to get 2 words minimum
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
            meaningful = [w for w in words if w.lower() not in stop_words][:2]
            if meaningful:
                title = " ".join([w.capitalize() for w in meaningful])
        
        return title if title and len(title.split()) >= 1 else "Untitled"
        
    except Exception as e:
        # Production error handling - return safe fallback
        words = text.split()[:8] if text else ["Untitled"]
        return " ".join(words).title() if words else "Untitled"


# ============================================================
# SECTION 5: TASK PROCESSING
# ============================================================
# Routes text to appropriate processing function based on
# task number or auto-detection using ML model.

def process_task_by_number(task_number, user_input):
    """Process task based on number selection - production-ready"""
    if user_input is None:
        user_input = ""
    if not isinstance(user_input, str):
        user_input = str(user_input)
    
    if not user_input.strip():
        return "Error: Empty input provided.", None
    
    # Input size validation (prevent DoS)
    MAX_INPUT_SIZE = 200000  # 200KB max
    if len(user_input) > MAX_INPUT_SIZE:
        return f"Error: Input too large (max {MAX_INPUT_SIZE} characters).", None
    
    task_number = str(task_number).strip()
    extracted_text = extract_content_from_prompt(user_input, task_number)
    
    task_map = {
        "1": ("summarization", summarize),
        "2": ("email", format_email),
        "3": ("paraphrasing", paraphrase),
        "4": ("title", generate_title),
    }
    
    if task_number not in task_map:
        return f"Error: Invalid task number '{task_number}'. Please choose 1, 2, 3, or 4.", None
    
    task_name, task_function = task_map[task_number]
    
    try:
        result = task_function(extracted_text)
        if not result or not isinstance(result, str):
            return f"Error: {task_name} returned invalid result.", task_name
        return result, task_name
    except Exception as e:
        return f"Error processing {task_name}: {str(e)}", task_name


def route_task(model, user_input, threshold=0.60):
    """Auto-detect task type using ML model - production-ready"""
    if model is None:
        return "Error: Model not available.", None, 0.0
    
    if user_input is None:
        user_input = ""
    if not isinstance(user_input, str):
        user_input = str(user_input)
    
    if not user_input.strip():
        return "Error: Empty input provided.", None, 0.0
    
    # Input size validation (prevent DoS)
    MAX_INPUT_SIZE = 200000  # 200KB max
    if len(user_input) > MAX_INPUT_SIZE:
        return f"Error: Input too large (max {MAX_INPUT_SIZE} characters).", None, 0.0
    
    if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
        threshold = 0.60

    try:
        cleaned = clean_text(user_input)
        
        if not cleaned.strip():
            return "Error: No valid text after cleaning.", None, 0.0

        probs = model.predict_proba([cleaned])[0]
        classes = model.classes_

        if len(probs) == 0 or len(classes) == 0:
            return "Error: Model prediction failed.", None, 0.0

        max_prob = max(probs)
        predicted_class = classes[probs.argmax()]

        if max_prob < threshold:
            return "Uncertain request. Please clarify your intent.", predicted_class, max_prob

        # Extract content from prompt before processing
        task_number_map = {
            "summarization": "1",
            "email": "2",
            "paraphrasing": "3",
            "title": "4"
        }
        
        task_num = task_number_map.get(predicted_class, None)
        if task_num:
            extracted_text = extract_content_from_prompt(user_input, task_num)
        else:
            extracted_text = user_input
        
        # Call appropriate function with error handling
        try:
            if predicted_class == "summarization":
                result = summarize(extracted_text)
            elif predicted_class == "email":
                result = format_email(extracted_text)
            elif predicted_class == "paraphrasing":
                result = paraphrase(extracted_text)
            elif predicted_class == "title":
                result = generate_title(extracted_text)
            else:
                return "Error: Unknown task class.", predicted_class, max_prob
            
            return result, predicted_class, max_prob
            
        except Exception as func_error:
            return f"Error processing {predicted_class}: {str(func_error)}", predicted_class, max_prob
            
    except Exception as e:
        return f"Error during task routing: {str(e)}", None, 0.0


def process_text_direct(text, task_type=None, threshold=0.60):
    """Direct function to process text without API"""
    global model
    
    if model is None:
        return {
            "output": "",
            "predicted_task": None,
            "confidence": 0.0,
            "success": False,
            "error": "Model not loaded"
        }
    
    if not text or not str(text).strip():
        return {
            "output": "",
            "predicted_task": None,
            "confidence": 0.0,
            "success": False,
            "error": "Empty text provided"
        }
    
    try:
        if task_type and task_type.strip() and task_type in ["1", "2", "3", "4"]:
            output, task = process_task_by_number(task_type, text)
            return {
                "output": output,
                "predicted_task": task,
                "confidence": 1.0,
                "success": True,
                "error": None
            }
        else:
            output, task, confidence = route_task(model, text, threshold=threshold)
            return {
                "output": output,
                "predicted_task": task,
                "confidence": float(confidence) if confidence is not None else 0.0,
                "success": True,
                "error": None
            }
    except Exception as e:
        return {
            "output": "",
            "predicted_task": None,
            "confidence": 0.0,
            "success": False,
            "error": str(e)
        }


# ============================================================
# SECTION 6: HTTP SERVER FOR CHATBOT INTERFACE
# ============================================================
# Simple HTTP server that serves the chatbot HTML page and
# handles API requests for text processing.

# Global model variable
model = None

# Handle model path for both regular execution and PyInstaller executable
def get_model_path():
    """Get the correct model path for both regular execution and PyInstaller"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable (PyInstaller)
        base_path = sys._MEIPASS
    else:
        # Running as regular Python script
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(base_path, "rf_task_classifier.pkl")
    
    # Also check current directory (for development)
    if not os.path.exists(model_path):
        current_dir_model = os.path.join(os.getcwd(), "rf_task_classifier.pkl")
        if os.path.exists(current_dir_model):
            return current_dir_model
    
    return model_path

MODEL_PATH = get_model_path()

# Chatbot HTML content (embedded in code)
CHATBOT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Text Assistant - Chatbot</title>
    <style>
        :root {
            /* Light Mode Colors */
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-tertiary: #f0f0f0;
            --text-primary: #111111;
            --text-secondary: #666666;
            --text-tertiary: #999999;
            --border-color: #e9ecef;
            --border-color-strong: #dee2e6;
            --input-bg: #f8f9fa;
            --input-bg-focus: #ffffff;
            --button-primary: #007bff;
            --button-primary-hover: #0056b3;
            --button-primary-active: #004085;
            --message-user-bg: #007bff;
            --message-user-text: #ffffff;
            --message-bot-bg: #28a745;
            --message-bot-text: #ffffff;
            --shadow: rgba(0, 0, 0, 0.05);
            --shadow-strong: rgba(0, 0, 0, 0.08);
            --copy-button-bg: rgba(0, 0, 0, 0.05);
            --copy-button-hover: rgba(0, 0, 0, 0.1);
            --task-badge-bg: #f0f0f0;
            --task-badge-text: #666666;
        }

        [data-theme="dark"] {
            /* Dark Mode Colors */
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --bg-tertiary: #3a3a3a;
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --text-tertiary: #888888;
            --border-color: #404040;
            --border-color-strong: #505050;
            --input-bg: #2d2d2d;
            --input-bg-focus: #3a3a3a;
            --button-primary: #007bff;
            --button-primary-hover: #0056b3;
            --button-primary-active: #004085;
            --message-user-bg: #007bff;
            --message-user-text: #ffffff;
            --message-bot-bg: #28a745;
            --message-bot-text: #ffffff;
            --shadow: rgba(0, 0, 0, 0.3);
            --shadow-strong: rgba(0, 0, 0, 0.5);
            --copy-button-bg: rgba(255, 255, 255, 0.1);
            --copy-button-hover: rgba(255, 255, 255, 0.2);
            --task-badge-bg: #3a3a3a;
            --task-badge-text: #b0b0b0;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            margin: 0;
            overflow: hidden;
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        .chatbot-container {
            width: 100%;
            height: 100vh;
            background: var(--bg-primary);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chatbot-header {
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: clamp(12px, 3vw, 16px) clamp(12px, 4vw, 20px);
            border-bottom: 1px solid var(--border-color);
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header-content {
            display: flex;
            align-items: center;
            gap: clamp(8px, 2vw, 12px);
        }

        .header-icon {
            width: clamp(32px, 8vw, 40px);
            height: clamp(32px, 8vw, 40px);
            background: var(--bg-tertiary);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: clamp(16px, 4vw, 20px);
            flex-shrink: 0;
        }

        .header-text h1 {
            font-size: clamp(14px, 3.5vw, 18px);
            font-weight: 600;
            color: var(--text-primary);
            margin: 0;
            line-height: 1.2;
        }

        .header-text p {
            font-size: clamp(11px, 2.5vw, 13px);
            color: var(--text-secondary);
            margin: 0;
        }

        .theme-toggle {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 50%;
            width: clamp(36px, 9vw, 40px);
            height: clamp(36px, 9vw, 40px);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
            flex-shrink: 0;
            padding: 0;
        }

        .theme-toggle:hover {
            background: var(--border-color);
            transform: scale(1.05);
        }

        .theme-toggle:active {
            transform: scale(0.95);
        }

        .theme-toggle svg {
            width: clamp(18px, 4.5vw, 20px);
            height: clamp(18px, 4.5vw, 20px);
            color: var(--text-primary);
            transition: transform 0.3s ease;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: clamp(12px, 3vw, 20px) clamp(12px, 3vw, 16px);
            background: var(--bg-secondary);
            scroll-behavior: smooth;
        }

        /* Hide scrollbars but keep scrolling */
        .chat-messages::-webkit-scrollbar {
            display: none;
        }

        .chat-messages {
            -ms-overflow-style: none;
            scrollbar-width: none;
        }

        body {
            -ms-overflow-style: none;
            scrollbar-width: none;
        }

        body::-webkit-scrollbar {
            display: none;
        }

        .message {
            margin-bottom: clamp(8px, 2vw, 12px);
            display: flex;
            flex-direction: column;
            animation: fadeIn 0.2s ease-in;
            width: 100%;
            clear: both;
            position: relative;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message.user {
            align-items: flex-end;
            justify-content: flex-end;
        }

        .message.bot {
            align-items: flex-start;
            justify-content: flex-start;
        }

        .message-bubble {
            max-width: 85%;
            min-width: 60px;
            padding: clamp(8px, 2vw, 10px) clamp(10px, 2.5vw, 14px);
            border-radius: clamp(12px, 3vw, 18px);
            word-wrap: break-word;
            word-break: break-word;
            line-height: 1.5;
            white-space: pre-wrap;
            font-size: clamp(13px, 3.5vw, 15px);
            box-shadow: 0 1px 2px var(--shadow);
            display: inline-block;
            position: relative;
        }

        .copy-button {
            position: absolute;
            top: clamp(4px, 1vw, 6px);
            right: clamp(4px, 1vw, 6px);
            background: var(--copy-button-bg);
            border: none;
            border-radius: clamp(4px, 1vw, 6px);
            width: clamp(24px, 6vw, 28px);
            height: clamp(24px, 6vw, 28px);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.2s, background 0.2s;
            padding: 0;
            z-index: 10;
        }

        .message:hover .copy-button {
            opacity: 1;
        }

        .copy-button:hover {
            background: var(--copy-button-hover);
        }

        .copy-button:active {
            background: var(--copy-button-hover);
            opacity: 0.8;
        }

        .copy-button.copied {
            background: rgba(76, 175, 80, 0.2);
            opacity: 1;
        }

        .copy-icon {
            width: clamp(14px, 3.5vw, 16px);
            height: clamp(14px, 3.5vw, 16px);
            color: var(--text-secondary);
        }

        .message.user .copy-button {
            background: rgba(255, 255, 255, 0.2);
        }

        .message.user .copy-button:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        .message.user .copy-icon {
            color: #ffffff;
        }

        .message.bot .copy-button {
            background: rgba(255, 255, 255, 0.2);
        }

        .message.bot .copy-button:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        .message.bot .copy-icon {
            color: #ffffff;
        }

        .message.user .message-bubble {
            background: var(--message-user-bg);
            color: var(--message-user-text);
            border-bottom-right-radius: 4px;
            margin-right: 0;
            margin-left: auto;
        }

        .message.bot .message-bubble {
            background: var(--message-bot-bg);
            color: var(--message-bot-text);
            border: none;
            border-bottom-left-radius: 4px;
            box-shadow: 0 1px 2px var(--shadow-strong);
            margin-left: 0;
            margin-right: auto;
        }

        .message-info {
            font-size: clamp(9px, 2.2vw, 11px);
            color: var(--text-tertiary);
            margin-top: clamp(2px, 0.5vw, 4px);
            padding: 0 clamp(2px, 0.5vw, 4px);
            width: 100%;
            display: flex;
            align-items: center;
        }

        .message.user .message-info {
            justify-content: flex-end;
            padding-right: clamp(2px, 0.5vw, 4px);
        }

        .message.bot .message-info {
            justify-content: flex-start;
            padding-left: clamp(2px, 0.5vw, 4px);
        }

        .task-badge {
            display: inline-block;
            padding: clamp(2px, 0.5vw, 3px) clamp(6px, 1.5vw, 8px);
            border-radius: clamp(8px, 2vw, 10px);
            font-size: clamp(9px, 2.2vw, 10px);
            font-weight: 500;
            margin-left: clamp(4px, 1vw, 6px);
            text-transform: uppercase;
            background: var(--task-badge-bg);
            color: var(--task-badge-text);
        }

        .task-summarization {
            background: #e3f2fd;
            color: #1976d2;
        }

        [data-theme="dark"] .task-summarization {
            background: #1e3a5f;
            color: #64b5f6;
        }

        .task-email {
            background: #f3e5f5;
            color: #7b1fa2;
        }

        [data-theme="dark"] .task-email {
            background: #4a2c5a;
            color: #ba68c8;
        }

        .task-paraphrasing {
            background: #e8f5e9;
            color: #388e3c;
        }

        [data-theme="dark"] .task-paraphrasing {
            background: #2e4a2f;
            color: #66bb6a;
        }

        .task-title {
            background: #fff3e0;
            color: #f57c00;
        }

        [data-theme="dark"] .task-title {
            background: #4a3a2a;
            color: #ffb74d;
        }

        .confidence-bar {
            width: 100%;
            height: 3px;
            background: var(--border-color);
            border-radius: 2px;
            margin-top: 4px;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            background: #28a745;
            transition: width 0.3s ease;
        }

        .input-area {
            padding: clamp(8px, 2vw, 12px) clamp(12px, 3vw, 16px);
            background: var(--bg-primary);
            border-top: 1px solid var(--border-color);
            flex-shrink: 0;
        }

        .input-container {
            display: flex;
            gap: clamp(6px, 1.5vw, 8px);
            align-items: flex-end;
            flex-wrap: wrap;
        }

        .task-selector {
            padding: clamp(8px, 2vw, 10px) clamp(10px, 2.5vw, 12px);
            border: 1px solid var(--border-color-strong);
            border-radius: clamp(16px, 4vw, 20px);
            font-size: clamp(11px, 2.8vw, 13px);
            background: var(--bg-primary);
            cursor: pointer;
            min-width: clamp(90px, 22vw, 110px);
            flex: 0 0 auto;
            color: var(--text-primary);
            width: 100%;
        }

        .task-selector:focus {
            outline: none;
            border-color: var(--button-primary);
        }

        .text-input-wrapper {
            display: flex;
            gap: clamp(6px, 1.5vw, 8px);
            align-items: flex-end;
            flex: 1 1 100%;
            width: 100%;
        }

        .text-input {
            flex: 1;
            padding: clamp(8px, 2vw, 10px) clamp(12px, 3vw, 16px);
            border: 1px solid var(--border-color-strong);
            border-radius: clamp(20px, 5vw, 24px);
            font-size: clamp(13px, 3.5vw, 15px);
            font-family: inherit;
            resize: none;
            min-height: clamp(40px, 10vw, 44px);
            max-height: clamp(100px, 25vw, 120px);
            background: var(--input-bg);
            color: var(--text-primary);
            transition: background 0.3s ease, border-color 0.3s ease;
        }

        .text-input:focus {
            outline: none;
            border-color: var(--button-primary);
            background: var(--input-bg-focus);
        }

        .text-input::selection {
            background: #fff3cd;
            color: #856404;
        }

        [data-theme="dark"] .text-input::selection {
            background: #4a5568;
            color: #e0e0e0;
        }

        .text-input::-moz-selection {
            background: #fff3cd;
            color: #856404;
        }

        [data-theme="dark"] .text-input::-moz-selection {
            background: #4a5568;
            color: #e0e0e0;
        }

        .send-button {
            padding: 0;
            background: var(--button-primary);
            color: white;
            border: none;
            border-radius: 50%;
            font-size: clamp(13px, 3.5vw, 15px);
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s, transform 0.2s;
            flex: 0 0 auto;
            width: clamp(40px, 10vw, 44px);
            height: clamp(40px, 10vw, 44px);
            align-self: flex-end;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .send-icon {
            width: clamp(18px, 4.5vw, 20px);
            height: clamp(18px, 4.5vw, 20px);
            color: white;
        }

        .send-button:hover {
            background: var(--button-primary-hover);
        }

        .send-button:active {
            background: var(--button-primary-active);
        }

        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: var(--text-tertiary);
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .quick-actions {
            display: flex;
            gap: clamp(4px, 1vw, 6px);
            margin-bottom: clamp(8px, 2vw, 10px);
            flex-wrap: wrap;
        }

        .quick-action-btn {
            padding: clamp(6px, 1.5vw, 8px) clamp(10px, 2.5vw, 14px);
            background: var(--bg-secondary);
            border: 1px solid var(--border-color-strong);
            border-radius: clamp(14px, 3.5vw, 18px);
            font-size: clamp(10px, 2.5vw, 12px);
            cursor: pointer;
            transition: all 0.2s;
            flex: 1 1 calc(50% - 3px);
            color: var(--text-primary);
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }

        .action-icon {
            width: 14px;
            height: 14px;
            flex-shrink: 0;
        }

        .quick-action-btn span {
            line-height: 1;
        }

        .header-icon svg {
            width: clamp(20px, 5vw, 24px);
            height: clamp(20px, 5vw, 24px);
            color: var(--text-primary);
        }

        .quick-action-btn:hover {
            background: var(--bg-tertiary);
            border-color: var(--border-color);
        }

        .quick-action-btn:active {
            background: var(--border-color);
        }

        /* Responsive Media Queries */
        @media (max-width: 400px) {
            .message-bubble {
                max-width: 90%;
                font-size: 13px;
            }
            
            .header-text p {
                display: none;
            }
            
            .input-container {
                flex-direction: column;
            }
            
            .text-input-wrapper {
                width: 100%;
            }
        }

        @media (min-width: 401px) and (max-width: 500px) {
            .message-bubble {
                max-width: 85%;
            }
        }

        @media (min-width: 501px) {
            .message-bubble {
                max-width: 80%;
            }
            
            .input-container {
                flex-direction: row;
            }
            
            .text-input-wrapper {
                flex-direction: row;
            }
        }

    </style>
</head>
<body>
    <div class="chatbot-container">
        <div class="chatbot-header">
            <div class="header-content">
                <div class="header-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
                        <path fill="currentColor" d="M17 3a1 1 0 1 0-2 0v1h-4.75A3.25 3.25 0 0 0 7 7.25v5.5A3.25 3.25 0 0 0 10.25 16h11.5A3.25 3.25 0 0 0 25 12.75v-5.5A3.25 3.25 0 0 0 21.75 4H17zM8.25 19A3.25 3.25 0 0 0 5 22.25v.45c0 2.17 1.077 4.043 3.013 5.332C9.92 29.302 12.634 30 16 30s6.08-.698 7.987-1.968C25.923 26.742 27 24.871 27 22.7v-.45A3.25 3.25 0 0 0 23.75 19zm4.25-7.25a1.75 1.75 0 1 1 0-3.5a1.75 1.75 0 0 1 0 3.5M21.25 10a1.75 1.75 0 1 1-3.5 0a1.75 1.75 0 0 1 3.5 0"/>
                    </svg>
                </div>
                <div class="header-text">
                    <h1>ML Text Assistant</h1>
                    <p>AI-Powered Text Processing</p>
                </div>
            </div>
            <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()" title="Toggle theme">
                <svg id="themeIcon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="5"></circle>
                    <line x1="12" y1="1" x2="12" y2="3"></line>
                    <line x1="12" y1="21" x2="12" y2="23"></line>
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                    <line x1="1" y1="12" x2="3" y2="12"></line>
                    <line x1="21" y1="12" x2="23" y2="12"></line>
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                </svg>
            </button>
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-bubble">
                    Hi! I'm your AI text assistant. I can help you with:
                    <br><br>
                    • <strong>Summarization</strong> - Condense text into brief summaries<br>
                    • <strong>Email Formatting</strong> - Convert text to professional emails<br>
                    • <strong>Paraphrasing</strong> - Reword text while keeping the meaning<br>
                    • <strong>Title Generation</strong> - Create catchy titles from content<br><br>
                    Type your text or use the quick actions below to get started!
                </div>
            </div>
        </div>

        <div class="input-area">
            <div class="quick-actions">
                <button class="quick-action-btn" onclick="setQuickAction('summarize')">
                    <svg class="action-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z" fill="currentColor"/>
                    </svg>
                    <span>Summarize</span>
                </button>
                <button class="quick-action-btn" onclick="setQuickAction('email')">
                    <svg class="action-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z" fill="currentColor"/>
                    </svg>
                    <span>Email</span>
                </button>
                <button class="quick-action-btn" onclick="setQuickAction('paraphrase')">
                    <svg class="action-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z" fill="currentColor"/>
                    </svg>
                    <span>Paraphrase</span>
                </button>
                <button class="quick-action-btn" onclick="setQuickAction('title')">
                    <svg class="action-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M5 4v3h5.5v12h3V7H19V4H5z" fill="currentColor"/>
                    </svg>
                    <span>Title</span>
                </button>
            </div>
            <div class="input-container">
                <div class="text-input-wrapper">
                    <textarea 
                        class="text-input" 
                        id="textInput" 
                        placeholder="Select the quick actions above and type your text here..."
                        rows="2"
                    ></textarea>
                    <button class="send-button" id="sendButton" onclick="sendMessage()">
                        <svg class="send-icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                            <path fill="currentColor" fill-rule="evenodd" d="M3.402 6.673c-.26-2.334 2.143-4.048 4.266-3.042l11.944 5.658c2.288 1.083 2.288 4.339 0 5.422L7.668 20.37c-2.123 1.006-4.525-.708-4.266-3.042L3.882 13H12a1 1 0 1 0 0-2H3.883z" clip-rule="evenodd"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_URL = 'http://127.0.0.1:8000';
        const chatMessages = document.getElementById('chatMessages');
        const textInput = document.getElementById('textInput');
        const sendButton = document.getElementById('sendButton');
        const taskSelector = document.getElementById('taskSelector');
        const themeToggle = document.getElementById('themeToggle');
        const themeIcon = document.getElementById('themeIcon');

        // Theme management
        function getTheme() {
            const savedTheme = localStorage.getItem('theme');
            return savedTheme || 'light';
        }

        function setTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
            updateThemeIcon(theme);
        }

        function updateThemeIcon(theme) {
            if (theme === 'dark') {
                themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
            } else {
                themeIcon.innerHTML = '<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>';
            }
        }

        function toggleTheme() {
            const currentTheme = getTheme();
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            setTheme(newTheme);
        }

        // Initialize theme on page load
        setTheme(getTheme());

        async function copyToClipboard(text, button) {
            try {
                await navigator.clipboard.writeText(text);
                
                // Visual feedback
                const originalHTML = button.innerHTML;
                button.classList.add('copied');
                button.innerHTML = '<svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
                button.title = 'Copied!';
                
                // Reset after 2 seconds
                setTimeout(() => {
                    button.classList.remove('copied');
                    button.innerHTML = originalHTML;
                    button.title = 'Copy to clipboard';
                }, 2000);
            } catch (err) {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                
                try {
                    document.execCommand('copy');
                    const originalHTML = button.innerHTML;
                    button.classList.add('copied');
                    button.innerHTML = '<svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
                    button.title = 'Copied!';
                    
                    setTimeout(() => {
                        button.classList.remove('copied');
                        button.innerHTML = originalHTML;
                        button.title = 'Copy to clipboard';
                    }, 2000);
                } catch (err) {
                    alert('Failed to copy text');
                }
                
                document.body.removeChild(textArea);
            }
        }

        function setQuickAction(action) {
            const prefixes = {
                'summarize': 'Summarize this paragraph: ',
                'email': 'Rewrite this as a professional email: ',
                'paraphrase': 'Paraphrase this sentence: ',
                'title': 'Generate a title for: '
            };
            
            const prefix = prefixes[action] || '';
            
            // Always clear the field and set the new prefix
            textInput.value = prefix;
            
            // Focus and select the prefix text for highlighting
            textInput.focus();
            const prefixLength = prefix.length;
            
            // Select the prefix text to highlight it
            if (textInput.setSelectionRange) {
                textInput.setSelectionRange(0, prefixLength);
            } else if (textInput.createTextRange) {
                const range = textInput.createTextRange();
                range.collapse(true);
                range.moveEnd('character', prefixLength);
                range.moveStart('character', 0);
                range.select();
            }
            
            // Move cursor to end after a brief moment to allow user to see the highlight
            setTimeout(() => {
                textInput.setSelectionRange(textInput.value.length, textInput.value.length);
            }, 500);
        }

        function addMessage(text, isUser, task = null, confidence = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            bubble.textContent = text;
            
            // Add copy button
            const copyButton = document.createElement('button');
            copyButton.className = 'copy-button';
            copyButton.title = 'Copy to clipboard';
            copyButton.innerHTML = '<svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
            copyButton.onclick = () => copyToClipboard(text, copyButton);
            
            messageDiv.appendChild(bubble);
            messageDiv.appendChild(copyButton);
            
            if (!isUser && task && confidence !== null) {
                const info = document.createElement('div');
                info.className = 'message-info';
                
                const taskBadge = document.createElement('span');
                taskBadge.className = `task-badge task-${task}`;
                taskBadge.textContent = task;
                info.appendChild(taskBadge);
                
                const confidenceText = document.createElement('span');
                confidenceText.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
                info.appendChild(confidenceText);
                
                messageDiv.appendChild(info);
                
                const confidenceBar = document.createElement('div');
                confidenceBar.className = 'confidence-bar';
                const confidenceFill = document.createElement('div');
                confidenceFill.className = 'confidence-fill';
                confidenceFill.style.width = `${confidence * 100}%`;
                confidenceBar.appendChild(confidenceFill);
                messageDiv.appendChild(confidenceBar);
            }
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        async function sendMessage() {
            const text = textInput.value.trim();
            if (!text) return;
            
            addMessage(text, true);
            textInput.value = '';
            
            sendButton.disabled = true;
            sendButton.innerHTML = '<div class="loading"></div>';
            
            // Detect task type from prefix
            let taskType = null;
            const lowerText = text.toLowerCase();
            if (lowerText.startsWith('summarize this paragraph:') || lowerText.startsWith('summarize this')) {
                taskType = '1';
            } else if (lowerText.startsWith('rewrite this as a professional email:') || lowerText.startsWith('rewrite this as')) {
                taskType = '2';
            } else if (lowerText.startsWith('paraphrase this sentence:') || lowerText.startsWith('paraphrase this') || 
                       lowerText.startsWith('paraphrase:') || lowerText.startsWith('reword:') || 
                       lowerText.startsWith('rewrite:') || lowerText.startsWith('rephrase:')) {
                taskType = '3';
            } else if (lowerText.startsWith('generate a title for:') || lowerText.startsWith('generate a title') ||
                       lowerText.startsWith('create a title') || lowerText.startsWith('suggest a title')) {
                taskType = '4';
            }
            
            try {
                const response = await fetch(`${API_URL}/process`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        task_type: taskType,
                        threshold: 0.60
                    })
                });

                const data = await response.json();
                
                if (data.success) {
                    addMessage(data.output, false, data.predicted_task, data.confidence);
                } else {
                    addMessage(`❌ Error: ${data.error || 'Unknown error'}`, false);
                }
            } catch (error) {
                addMessage(`❌ Connection error: ${error.message}`, false);
            } finally {
                sendButton.disabled = false;
                sendButton.innerHTML = '<svg class="send-icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M3.402 6.673c-.26-2.334 2.143-4.048 4.266-3.042l11.944 5.658c2.288 1.083 2.288 4.339 0 5.422L7.668 20.37c-2.123 1.006-4.525-.708-4.266-3.042L3.882 13H12a1 1 0 1 0 0-2H3.883z" clip-rule="evenodd"/></svg>';
            }
        }

        textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        textInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
    </script>
</body>
</html>"""


class ChatbotHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for chatbot interface"""
    
    def do_GET(self):
        """Handle GET requests - serve chatbot HTML page"""
        if self.path == '/' or self.path == '/chatbot':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(CHATBOT_HTML.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests for text processing"""
        if self.path == '/process':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                text = data.get('text', '')
                task_type = data.get('task_type', None)
                threshold = data.get('threshold', 0.60)
                
                # Process text directly
                result = process_text_direct(text, task_type, threshold)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                error_response = {
                    "output": "",
                    "predicted_task": None,
                    "confidence": 0.0,
                    "success": False,
                    "error": str(e)
                }
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


# ============================================================
# SECTION 7: APPLICATION INITIALIZATION
# ============================================================
# Loads or trains the ML model and starts the HTTP server
# before opening the embedded browser window.

def load_model():
    """Load or train the ML model"""
    global model, MODEL_PATH
    try:
        # Update model path in case it changed
        MODEL_PATH = get_model_path()
        
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print("Model not found. Training new model...")
            print("This may take a few moments...")
            df = generate_dataset(samples_per_class=250)
            model = train_model(df)
            print("Model trained and ready!")
            # Try to save model in current directory if running as exe
            try:
                save_path = os.path.join(os.getcwd(), "rf_task_classifier.pkl")
                joblib.dump(model, save_path)
                print(f"Model saved to {save_path}")
            except Exception as save_error:
                print(f"Note: Could not save model: {save_error}")
    except Exception as e:
        print(f"Error loading/training model: {str(e)}")
        raise


def start_server(port=8000):
    """Start HTTP server in background thread"""
    def run_server():
        try:
            print("Starting HTTP server...")
            server = HTTPServer(('127.0.0.1', port), ChatbotHandler)
            print(f"Server started on http://127.0.0.1:{port}")
            server.serve_forever()
        except Exception as e:
            print(f"Error starting server: {str(e)}")
            raise
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    # Give server time to start
    import time
    time.sleep(0.5)
    return server_thread


# ============================================================
# SECTION 8: MAIN EXECUTION
# ============================================================
# Main entry point that initializes the application and opens
# the embedded browser window automatically.

if __name__ == "__main__":
    # Check if pywebview is available
    if not WEBVIEW_AVAILABLE:
        print("ERROR: pywebview is not installed!")
        print("Please install it with: pip install pywebview")
        sys.exit(1)
    
    # Load or train model
    print("Initializing ML Text Assistant...")
    load_model()
    
    # Start HTTP server
    start_server(8000)
    
    # URL to load in embedded browser
    chatbot_url = "http://127.0.0.1:8000/chatbot"
    
    # Create and open embedded browser window
    # Window opens automatically when webview.start() is called
    # Portrait orientation with 9:16 aspect ratio (405:720)
    print("Opening embedded browser window...")
    
    try:
        webview.create_window(
            title="Multi-Function AI Text Assistant",  # Window title
            url=chatbot_url,                           # URL to load
            width=405,                                 # Window width (portrait - slightly bigger)
            height=720,                                # Window height (9:16 ratio)
            resizable=True,                            # Allow window resizing
            min_size=(270, 480)                        # Minimum window size (maintains 9:16 ratio)
        )
        
        # Start the webview event loop (blocks until window is closed)
        # This automatically opens the window when called
        webview.start(debug=False)
    except Exception as e:
        # Fallback if webview fails (e.g., missing GTK/QT on Linux)
        error_msg = str(e)
        print("\n" + "="*60)
        print("WARNING: Could not open embedded browser window!")
        print("="*60)
        print(f"Error: {error_msg}")
        print("\nFalling back to system browser...")
        print("="*60)
        print(f"\nThe application is running at: {chatbot_url}")
        print("\nPlease open this URL in your web browser manually.")
        print("\nTo fix this issue, install the required GUI libraries:")
        print("\n  For Linux (GTK):")
        print("    Ubuntu/Debian:")
        print("      sudo apt-get update")
        print("      sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2-4.1")
        print("      # OR: sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2*")
        print("      # OR: sudo apt-get install python3-gi python3-gi-cairo libwebkit2gtk-4.0-dev")
        print("    Fedora/RHEL:   sudo yum install python3-gobject webkitgtk4")
        print("\n  For Linux (QT):")
        print("    Ubuntu/Debian: sudo apt-get install python3-pyqt5 python3-pyqt5.qtwebengine")
        print("    Fedora/RHEL:   sudo yum install python3-qt5")
        print("\n" + "="*60)
        print("\nServer is running. Press Ctrl+C to stop.")
        print("="*60 + "\n")
        
        # Try to open in system browser as fallback
        try:
            import webbrowser
            import time
            time.sleep(1)  # Give server a moment to start
            webbrowser.open(chatbot_url)
            print(f"Opened {chatbot_url} in your default browser.\n")
        except Exception as browser_error:
            print(f"Could not open browser automatically: {browser_error}\n")
        
        # Keep server running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            sys.exit(0)
