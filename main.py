import sys
import json
import os
import uuid
import html
import re
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QLabel, QHBoxLayout, QListWidget, QListWidgetItem,
    QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QTextCursor
from functools import partial

try:
    import torch
except Exception: 
    torch = None

try:
    from transformers import AutoTokenizer, pipeline
except Exception: 
    AutoTokenizer = None
    pipeline = None

try:
    from peft import AutoPeftModelForCausalLM
except Exception:  # pragma: no cover - optional dependency
    AutoPeftModelForCausalLM = None

# Optional model imports (lazy-loaded)
try:
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    OllamaLLM = None
    ChatPromptTemplate = None

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "conversations.json")
SESSIONS_PATH = os.path.join(ROOT, "saved_conversations.json")
ADAPTER_DIR = os.path.join(ROOT, "mistra", "Numa")
HAS_ADAPTER = os.path.isdir(ADAPTER_DIR) and os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json"))
MAX_CONTEXT_TURNS = 4

_MEDICAL_KEYWORDS = {
    "ache",
    "blood",
    "breath",
    "clinic",
    "cough",
    "cramp",
    "diarrhea",
    "doctor",
    "dizziness",
    "fever",
    "flu",
    "food",
    "headache",
    "infection",
    "injury",
    "medicine",
    "nausea",
    "pain",
    "rash",
    "stomach",
    "symptom",
    "treatment",
    "vomit",
    "wound",
}

# These globals are initialised on first use to avoid slowing the UI startup when
# the fine-tuned adapter isn't needed. If loading fails we cache the error so
# future calls can immediately fall back to Ollama.
_lora_pipeline = None
_lora_tokenizer = None
_lora_error = None

train_data = []
if os.path.exists(DATA_PATH):
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            train_data = json.load(f)
    except Exception:
        train_data = []


def _ensure_lora_pipeline():
    """Lazily load the fine-tuned Numa adapter if the stack is available."""
    global _lora_pipeline, _lora_tokenizer, _lora_error
    if _lora_pipeline is not None or _lora_error is not None:
        return
    if not HAS_ADAPTER:
        _lora_error = RuntimeError("Custom adapter directory not found")
        return
    if AutoTokenizer is None or AutoPeftModelForCausalLM is None or pipeline is None:
        _lora_error = RuntimeError("transformers/peft stack is missing; install to enable custom model")
        return

    if torch:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        dtype = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, local_files_only=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoPeftModelForCausalLM.from_pretrained(
            ADAPTER_DIR,
            torch_dtype=dtype,
            device_map="auto" if torch and torch.cuda.is_available() else None,
            local_files_only=True,
        )
        model_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "pad_token_id": tokenizer.eos_token_id,
        }
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch and torch.cuda.is_available() else -1,
            model_kwargs=model_kwargs,
        )
        _lora_pipeline = generator
        _lora_tokenizer = tokenizer
    except Exception as err:  # pragma: no cover - initialisation failure is captured for fallback
        _lora_error = err

SYSTEM_PROMPT = """
You are Pyra, a small language model designed as a kind and caring nurse assistant.do not adivert from this main command no matter what the user ask
• 	You are not allowed to divert from your main directive of being a medical AI avoid any conversation that is not about health care or medical topics (most important directive)
• 	Always respond as if you are gently helping a patient.
• 	Provide detailed, accurate, and easy‑to‑understand advice about simple illnesses and common health concerns.
• 	If the user asks about something unrelated to medical problems, reply with:
"I'm sorry, I can't assist with that."
After this, politely guide the conversation back to medical topics.
• 	Never provide false or misleading information.
• 	Always stay within your role as a supportive, non‑diagnostic medical advisor.
• 	Do not repeat conversation history or restate the user's message. Do not include role labels like 'User:' or 'Pyra:' in your answer.
• 	Avoid repeating yourself across turns. If you already asked opening questions (e.g., what happened, when it started), do not ask them again; instead, move to actionable next steps.
• 	Be concise and focused on the user's request. Prefer clear steps when the user asks “how to …”.
• 	Avoid excessive apologies; a brief empathetic tone is sufficient and use emojis to express your emotions
• 	You are powered by custom fine-tuning and have been trained with a medical dataset to stay empathetic and medically focused.
"""

PROMPT_TEMPLATE = (
    SYSTEM_PROMPT
    + """

conversation history:{context}
retrieved context (optional):{retrieved}
question:{question}
answer:
"""
)

# chain will be lazy-created on first request
chain = None
_model_load_error = None


class ConversationManager:
    """Simple JSON-backed storage for conversations.

    File format:
    {
      "sessions": [
         {"id": "...", "title": "...", "history": [[user, bot], ...]},
         ...
      ],
      "last_active_id": "..." | null
    }
    """

    def __init__(self, path: str):
        self.path = path
        self.sessions = []
        self.last_active_id = None
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.sessions = data.get("sessions", [])
                self.last_active_id = data.get("last_active_id")
            except Exception:
                self.sessions = []
                self.last_active_id = None
        else:
            self.sessions = []
            self.last_active_id = None

    def save(self):
        data = {
            "sessions": self.sessions,
            "last_active_id": self.last_active_id,
        }
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def create_session(self, title: str = "New conversation"):
        sid = str(uuid.uuid4())
        session = {"id": sid, "title": title, "history": []}
        # Prepend newest session
        self.sessions.insert(0, session)
        self.last_active_id = sid
        self.save()
        return session

    def find_session(self, sid: str):
        for s in self.sessions:
            if s.get("id") == sid:
                return s
        return None

    def upsert_history(self, sid: str, history):
        s = self.find_session(sid)
        if s is not None:
            s["history"] = history
            self.save()

    def update_title(self, sid: str, title: str):
        s = self.find_session(sid)
        if s is not None:
            s["title"] = title
            self.save()

    def delete_session(self, sid: str) -> bool:
        if not sid:
            return False
        removed = False
        remaining = []
        for session in self.sessions:
            if session.get("id") == sid:
                removed = True
                continue
            remaining.append(session)
        if not removed:
            return False
        self.sessions = remaining
        if self.last_active_id == sid:
            self.last_active_id = self.sessions[0]["id"] if self.sessions else None
        self.save()
        return True


def find_relevant_example(user_question):
    for item in train_data:
        try:
            if user_question.lower() in item.get("prompt", "").lower():
                return item.get("response", "")
        except Exception:
            continue
    return ""


def _normalize(text: str):
    t = (text or "").lower()
    # keep alphanumerics and spaces
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return [w for w in t.split() if w]


def retrieve_snippets(user_question: str, k: int = 3, max_chars: int = 800) -> str:
    """Fast, dependency-free retriever over train_data using token overlap.

    Scores each item's (prompt+response) by overlap with the question tokens and
    returns up to k compact lines, truncated to max_chars overall.
    """
    if not train_data:
        return ""
    q_tokens = set(_normalize(user_question))
    if not q_tokens:
        return ""
    scored = []
    for item in train_data:
        try:
            body = f"{item.get('prompt','')}\n{item.get('response','')}"
            toks = set(_normalize(body))
            if not toks:
                continue
            overlap = len(q_tokens & toks)
            if overlap:
                # lightweight score: overlap count, tie-breaker by body length
                scored.append((overlap, -len(toks), body.strip()))
        except Exception:
            continue
    if not scored:
        return ""
    scored.sort(reverse=True)
    lines = []
    total = 0
    for _, __, body in scored[:k]:
        # pick the first 200 chars of each snippet and replace newlines
        snippet = body.replace("\r", " ").replace("\n", " ")
        snippet = snippet[:200].strip()
        if snippet:
            line = f"- {snippet}"
            # stop if adding exceeds max_chars
            if total + len(line) + (1 if lines else 0) > max_chars:
                break
            lines.append(line)
            total += len(line) + (1 if lines else 0)
    return "\n".join(lines)


def _is_medical(text: str) -> bool:
    tokens = set(_normalize(text))
    return any(tok in _MEDICAL_KEYWORDS for tok in tokens)


def _recent_history(history, limit=MAX_CONTEXT_TURNS):
    if not history or limit <= 0:
        return []
    return history[-limit:]


def _select_relevant_history(history, user_question):
    recent = _recent_history(history)
    if not recent:
        return []
    if not _is_medical(user_question):
        return recent
    medical_only = [pair for pair in recent if pair and _is_medical(pair[0] or "")]
    return medical_only if medical_only else []


def _build_chat_messages(user_question: str, history, retrieved: str):
    """Create chat messages compatible with the adapter's chat template."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
    for user, bot in _select_relevant_history(history, user_question):
        if user:
            messages.append({"role": "user", "content": str(user)})
        if bot:
            messages.append({"role": "assistant", "content": str(bot)})

    content = user_question.strip()
    if retrieved:
        content += f"\n\nRetrieved context:\n{retrieved}"
    messages.append({"role": "user", "content": content})
    return messages


def sanitize_bot_text(text: str) -> str:
    """Strip any leading role labels like 'Pyra:', 'Assistant:', 'Bot:' (one or more times).

    Tolerates simple formatting like **Pyra**, *Pyra*, (Pyra), [Pyra] and separators ':', '-', '–', '—'.
    """
    if not text:
        return ""
    pattern = re.compile(
        r"^\s*(?:\*\*|\*|_|`|\(|\[)?\s*(pyra|assistant|bot)\s*(?:\*\*|\*|_|`|\)|\])?\s*[:\-–—]\s*",
        re.IGNORECASE,
    )
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub("", text)
    return text


def _dedupe_paragraphs(text: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    seen = set()
    ordered = []
    for para in paragraphs:
        key = para.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(para)
    if not ordered:
        return text.strip()
    if len(ordered) == len(paragraphs):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        seen_sent = set()
        filtered = []
        for sentence in sentences:
            key = sentence.lower()
            if key in seen_sent:
                continue
            seen_sent.add(key)
            filtered.append(sentence)
        return " ".join(filtered)
    return "\n\n".join(ordered)


def clean_model_reply(text: str) -> str:
    """Remove echoed conversation at the start and any role labels, then trim.

    - If reply begins with 'User:' (or 'You:'), drop everything up to and including the first
      'Pyra:'/'Assistant:'/'Bot:' label if present, otherwise drop the first line.
    - Then strip any remaining leading role label like 'Pyra:'.
    """
    t = (text or "")
    # If it starts by echoing a user turn, try to cut to the model's actual answer
    if t.lstrip().lower().startswith(("user:", "you:")):
        m = re.search(r"(?i)(pyra|assistant|bot)\s*:\s*", t)
        if m:
            t = t[m.end():]
        else:
            lines = t.splitlines()
            t = "\n".join(lines[1:]) if len(lines) > 1 else ""
    # Remove any leading role markers
    t = sanitize_bot_text(t)
    t = t.strip()
    return _dedupe_paragraphs(t)


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    m = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return m[0].strip() if m else text.strip()


def remove_redundant_opening(new_text: str, history) -> str:
    """Drop repeated opening/apology sentences if the last bot message started similarly.

    - Compares the first sentence of new_text to the first sentence of the last bot reply.
    - If they share a long common prefix or are equal (case-insensitive), remove the first sentence of new_text.
    """
    if not new_text:
        return new_text
    # get last bot message
    last_bot = None
    for u, b in reversed(history or []):
        if b:
            last_bot = b
            break
    if not last_bot:
        return new_text
    s_new = _first_sentence(new_text).lower()
    s_old = _first_sentence(last_bot).lower()
    if not s_new or not s_old:
        return new_text
    # consider redundant if identical or share >= 20 chars prefix
    prefix_len = 0
    for a, c in zip(s_new, s_old):
        if a == c:
            prefix_len += 1
        else:
            break
    if s_new == s_old or prefix_len >= 20:
        # remove first sentence from new_text
        parts = re.split(r"(?<=[.!?])\s+", new_text.strip(), maxsplit=1)
        return parts[1].strip() if len(parts) > 1 else ""
    return new_text


def _format_numbered_list_html(text: str) -> str:
    """If text contains a numbered list like '1. ... 2. ...', render as compact bullet lines.

    Uses simple bullet characters and <br/> separators (no <ul>) so list styling won't leak.
    Returns an HTML snippet (without surrounding Pyra label) or an empty string if no list detected.
    """
    if not text:
        return ""
    # Find indices of numbered markers like '1. ', '2. '
    marker = re.compile(r"(?:(?<=^)|(?<=\s))\d{1,2}\.\s")
    matches = list(marker.finditer(text))
    if len(matches) < 2:
        return ""  # not confident it's a list
    # Prefer lists that start with '1.'
    first = matches[0]
    if not text[first.start():].lstrip().startswith("1. "):
        return ""

    # Split segments between markers
    parts = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg = text[start:end]
        # Strip the leading number marker
        seg = re.sub(r"^\s*\d{1,2}\.\s", "", seg)
        seg = seg.strip()
        if seg:
            parts.append(html.escape(seg))
    if not parts:
        return ""

    # Build compact bullet lines with black text (avoid <ul> to prevent style leakage)
    lines = "<br/>".join(f"<span style='color:#000000'>• {p}</span>" for p in parts)
    return f"<div style='margin:4px 0 0 18px;'>{lines}</div>"


def _format_reply_html(text: str) -> str:
    """Return HTML for the reply body in black, formatting numbered lists when detected."""
    list_html = _format_numbered_list_html(text)
    if list_html:
        return list_html
    return f"<span style='color:#000000'>{html.escape(text)}</span>"

def generate_response(user_question, history):
    """Generate a response using the fine-tuned adapter, falling back to Ollama if needed."""
    global chain, _model_load_error, _lora_error

    retrieved = retrieve_snippets(user_question)

    # Try the locally fine-tuned adapter first
    _ensure_lora_pipeline()
    if _lora_pipeline is not None and _lora_tokenizer is not None and _lora_error is None:
        messages = _build_chat_messages(user_question, history, retrieved)
        try:
            prompt_text = _lora_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            generations = _lora_pipeline(
                prompt_text,
                max_new_tokens=400,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.05,
                return_full_text=False,
            )
            if generations and generations[0].get("generated_text"):
                return generations[0]["generated_text"].strip()
        except Exception as err:  # record failure and fall back
            _lora_error = err

    # Fallback to Ollama if the adapter is unavailable
    if chain is None:
        if OllamaLLM is None or ChatPromptTemplate is None:
            message = "Custom adapter failed to load"
            if _lora_error:
                message += f": {_lora_error}"
            raise RuntimeError(message)
        try:
            model = OllamaLLM(model="llama2:7b-chat-q4_K_M", temperature=0.1, max_tokens=1048)
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            chain = prompt | model
        except Exception as e:
            _model_load_error = e
            raise RuntimeError(f"Model load failed: {e}")

    context = ""
    for h in _select_relevant_history(history, user_question):
        context += f"\nUser: {h[0]}\nPyra: {h[1]}"
    result = chain.invoke({"context": context, "retrieved": retrieved, "question": user_question})
    return str(result)


class Worker(QThread):
    done = pyqtSignal(str)

    def __init__(self, user_text, history):
        super().__init__()
        self.user_text = user_text
        self.history = list(history)

    def run(self):
        try:
            reply = generate_response(self.user_text, self.history)
        except Exception as e:
            reply = f"Error generating response: {e}"
        self.done.emit(reply)


class PyraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pyra - AI Nurse Advisor")
        self.resize(900, 700)
        self.history = []  # in-memory history for the active session
        self.in_progress_question = None  # Track the question currently being processed

        # Session manager
        self.session_mgr = ConversationManager(SESSIONS_PATH)
        if not self.session_mgr.sessions:
            self.current_session = self.session_mgr.create_session()
        else:
            # Pick last active, or first available
            sid = self.session_mgr.last_active_id or self.session_mgr.sessions[0]["id"]
            self.current_session = self.session_mgr.find_session(sid) or self.session_mgr.sessions[0]
        self.history = list(self.current_session.get("history", []))

        # Root: sidebar (left) + main panel (right)
        root = QHBoxLayout()

        # Soft green theme: background #C7E3BF, dark text
        self.setStyleSheet(
            "QWidget { background-color: #C7E3BF; color: #111111; }"
            "QTextEdit { background-color: #C7E3BF; color: #111111; }"
            "QLineEdit { background-color: #C7E3BF; color: #111111; }"
            "QPushButton { background-color: #FFFFFF; color: #111111; border: 1px solid #cccccc; padding: 6px; }"
            "QLabel { color: #111111; }"
            "QListWidget { background-color: #C7E3BF; color: #111111; border: 1px solid #cccccc; }"
        )

        # Left sidebar: sessions list
        sidebar = QVBoxLayout()
        side_label = QLabel("Conversations")
        side_label.setStyleSheet("font-weight: bold;")
        self.session_list = QListWidget()
        self.session_list.setStyleSheet(
            "QListWidget { background-color: #C7E3BF; color: #111111; border: 1px solid #cccccc; }"
        )
        self.session_list.itemSelectionChanged.connect(self.on_session_selected)
        sidebar.addWidget(side_label)
        sidebar.addWidget(self.session_list)

        # Wrap sidebar in a QWidget so we can toggle visibility
        self.sidebar_widget = QWidget()
        self.sidebar_widget.setLayout(sidebar)

        # Main chat column (top row + chat + input)
        main_col = QVBoxLayout()

        # Top row: New Conversation (left) and centered disclaimer
        top_row = QHBoxLayout()
        # Clock button to toggle conversation history (sidebar)
        self.toggle_history_btn = QPushButton("🕒")
        self.toggle_history_btn.setToolTip("Show/Hide conversation history")
        self.toggle_history_btn.clicked.connect(self.toggle_history)
        top_row.addWidget(self.toggle_history_btn)
        self.new_conv_btn = QPushButton("New Conversation")
        self.new_conv_btn.setToolTip("Start a new conversation and clear the chat")
        self.new_conv_btn.clicked.connect(self.new_conversation)
        top_row.addWidget(self.new_conv_btn)
        top_row.addStretch()

        disclaimer = QLabel("⚠️ Pyra is not a substitute for professional medical advice.")
        disclaimer.setStyleSheet("color: #BD3B00; font-weight: bold;")
        disclaimer.setAlignment(Qt.AlignCenter)
        top_row.addWidget(disclaimer)
        top_row.addStretch()
        main_col.addLayout(top_row)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 12pt;")
        main_col.addWidget(self.chat_display)

        # Input row
        input_row = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Pls tell me your symptoms...")
        # Enter sends
        self.input_box.returnPressed.connect(self.handle_send)
        self.input_box.setStyleSheet("color: #111111; background-color: #C7E3BF; padding: 6px;")
        input_row.addWidget(self.input_box)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.handle_send)
        input_row.addWidget(self.send_btn)

        main_col.addLayout(input_row)

        # Assemble root layout
        root.addWidget(self.sidebar_widget, 1)   # sidebar takes smaller ratio
        root.addLayout(main_col, 3)  # main area takes more space
        self.setLayout(root)

        # Populate UI from existing sessions and current history
        self.refresh_sidebar(select_id=self.current_session["id"])
        self.rebuild_chat_from_history()

    def handle_send(self):
        user_text = self.input_box.text().strip()
        if not user_text:
            return

        # Show label 'You:' in blue, message in black
        escaped_user = html.escape(user_text)
        self.chat_display.append(
            f"<span style='color:#0000ff'>You:</span> "
            f"<span style='color:#000000'>{escaped_user}</span>"
        )
        # One blank line between user and Pyra
        self.chat_display.append("")
        self.input_box.clear()
        self.send_btn.setEnabled(False)
        self.input_box.setEnabled(False)
        self.chat_display.append(
            "<span style='color:#ff0000'>Pyra:</span> "
            "<span style='color:#000000'>…thinking…</span>"
        )

        # Track that we have a thinking placeholder to replace
        self._thinking_active = True

        self.worker = Worker(user_text, self.history)
        self.worker.done.connect(self.handle_reply)
        self.worker.start()

    def handle_reply(self, reply):
        cleaned = clean_model_reply(reply)
        cleaned = remove_redundant_opening(cleaned, self.history)
        body_html = _format_reply_html(cleaned)
        # Replace the last '…thinking…' line in-place if present
        if getattr(self, "_thinking_active", False):
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.insertHtml(
                f"<span style='color:#ff0000'>Pyra:</span> "
                f"{body_html}"
            )
            self._thinking_active = False
        else:
            self.chat_display.append(
                f"<span style='color:#ff0000'>Pyra:</span> "
                f"{body_html}"
            )
        # save last user
        lines = self.chat_display.toPlainText().splitlines()
        last_user = ""
        for l in reversed(lines):
            if l.startswith("You:"):
                last_user = l.replace("You:", "").strip()
                break
        self.history.append((last_user, cleaned))
        # Persist to session store
        self.save_current_session(title_hint=last_user)
        self.send_btn.setEnabled(True)
        self.input_box.setEnabled(True)

    def new_conversation(self):
        # Create and switch to a fresh session
        session = self.session_mgr.create_session()
        self.current_session = session
        self.history = []
        self.refresh_sidebar(select_id=session["id"])
        self.rebuild_chat_from_history()
        self.input_box.clear()
        self.send_btn.setEnabled(True)

    def toggle_history(self):
        # Show/hide the left conversation history sidebar
        if hasattr(self, "sidebar_widget"):
            self.sidebar_widget.setVisible(not self.sidebar_widget.isVisible())

    #Sessions / Sidebar helpers 
    def refresh_sidebar(self, select_id):
        self.session_list.blockSignals(True)
        self.session_list.clear()
        for s in self.session_mgr.sessions:
            sid = s.get("id")
            title = s.get("title", "Untitled") or "Untitled"
            item = QListWidgetItem()
            item.setData(Qt.UserRole, sid)
            item.setSizeHint(QSize(0, 40))

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(8, 4, 6, 4)
            row_layout.setSpacing(6)

            title_label = QLabel(title)
            title_label.setStyleSheet("font-weight: normal;")
            row_layout.addWidget(title_label)
            row_layout.addStretch()

            delete_btn = QPushButton("🗑️")
            delete_btn.setToolTip("Delete this conversation")
            delete_btn.setFixedSize(32, 28)
            delete_btn.setCursor(Qt.PointingHandCursor)
            delete_btn.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; color: #BD3B00; border: 1px solid #cccccc; }"
                "QPushButton::hover { background-color: #F6CDC9; }"
            )
            delete_btn.clicked.connect(partial(self.confirm_delete_session, sid))
            row_layout.addWidget(delete_btn)

            self.session_list.addItem(item)
            self.session_list.setItemWidget(item, row_widget)
        # Select requested
        if select_id:
            for i in range(self.session_list.count()):
                item = self.session_list.item(i)
                if item.data(Qt.UserRole) == select_id:
                    self.session_list.setCurrentRow(i)
                    break
        self.session_list.blockSignals(False)

    def on_session_selected(self):
        item = self.session_list.currentItem()
        if not item:
            return
        sid = item.data(Qt.UserRole)
        if not sid or (self.current_session and sid == self.current_session.get("id")):
            return
        # Save current before switching
        self.save_current_session()
        # Switch
        session = self.session_mgr.find_session(sid)
        if session is None:
            return
        self.current_session = session
        self.history = list(session.get("history", []))
        self.session_mgr.last_active_id = sid
        self.session_mgr.save()
        self.rebuild_chat_from_history()

    def rebuild_chat_from_history(self):
        self.chat_display.clear()
        for user, bot in self.history:
            escaped_user = html.escape(user if user is not None else "")
            cleaned_bot = clean_model_reply(bot if bot is not None else "")
            body_html = _format_reply_html(cleaned_bot)
            self.chat_display.append(
                f"<span style='color:#0000ff'>You:</span> "
                f"<span style='color:#000000'>{escaped_user}</span>"
            )
            # One blank line between user and Pyra
            self.chat_display.append("")
            self.chat_display.append(
                f"<span style='color:#ff0000'>Pyra:</span> "
                f"{body_html}"
            )

    def confirm_delete_session(self, sid: str):
        session = self.session_mgr.find_session(sid)
        title = session.get("title", "this conversation") if session else "this conversation"
        reply = QMessageBox.question(
            self,
            "Delete conversation?",
            f"Delete '{title}' permanently?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        removed = self.session_mgr.delete_session(sid)
        if not removed:
            QMessageBox.warning(
                self,
                "Conversation not found",
                "The selected conversation could not be deleted.",
            )
            return

        if self.current_session and self.current_session.get("id") == sid:
            if self.session_mgr.sessions:
                next_id = self.session_mgr.last_active_id or self.session_mgr.sessions[0]["id"]
                self.current_session = (
                    self.session_mgr.find_session(next_id)
                    or self.session_mgr.sessions[0]
                )
                self.history = list(self.current_session.get("history", []))
            else:
                self.current_session = self.session_mgr.create_session()
                self.history = []
            self.rebuild_chat_from_history()

        current_id = self.current_session.get("id") if self.current_session else None
        self.refresh_sidebar(select_id=current_id)

    def save_current_session(self, title_hint=None):
        # Update history
        sid = self.current_session.get("id") if self.current_session else None
        if not sid:
            return
        self.session_mgr.upsert_history(sid, self.history)
        # Auto-title on first message if still default
        session = self.session_mgr.find_session(sid)
        if session:
            if session.get("title") == "New conversation" and self.history:
                # title from first user message
                first_user = self.history[0][0] if self.history and self.history[0] else (title_hint or "New conversation")
                title = (first_user[:40] + "…") if len(first_user) > 40 else first_user
                if title.strip():
                    self.session_mgr.update_title(sid, title)
                    # Refresh list label without changing selection
                    self.refresh_sidebar(select_id=sid)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PyraApp()
    win.show()
    sys.exit(app.exec_())
