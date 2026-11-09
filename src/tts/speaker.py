"""Client-side Web Speech API generator."""






import json
from typing import Optional


def generate_web_speech_html(text: str, rate: float = 1.0, pitch: float = 1.0) -> str:




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




    html = generate_web_speech_html(text)
    if st_module is not None:
        import streamlit.components.v1 as components
        components.html(html, height=0, width=0)
    return html
