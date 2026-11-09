"""
TTS (Text-to-Speech) module for Streamlit and headless edge deployment.
Uses browser client-side Web Speech API (speechSynthesis) to deliver zero-overhead,
hardware-independent audio confirmation without requiring soundcards or ALSA/PulseAudio
in cloud container environments.
"""

import json
from typing import Optional


def generate_web_speech_html(text: str, rate: float = 1.0, pitch: float = 1.0) -> str:
    """
    Generates an embedded HTML snippet that triggers browser-native Web Speech API.
    Works seamlessly across all modern browsers (Chrome, Edge, Safari, Firefox).
    """
    safe_text = json.dumps(text)
    html_code = f"""
    <script>
    (function() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance({safe_text});
            utterance.rate = {rate};
            utterance.pitch = {pitch};
            utterance.lang = 'en-US';
            window.speechSynthesis.speak(utterance);
        }}
    }})();
    </script>
    """
    return html_code


def render_tts(text: str, st_module=None):
    """
    Renders the browser speech synthesis snippet in a Streamlit context.
    If streamlit is not available or st_module is None, returns the HTML string.
    """
    html = generate_web_speech_html(text)
    if st_module is not None:
        import streamlit.components.v1 as components
        components.html(html, height=0, width=0)
    return html
