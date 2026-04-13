"""MLB Toolbox — Feedback & Suggestions page."""

import threading

import streamlit as st
import streamlit.components.v1 as _stc

try:
    import requests as _requests
    _requests_available = True
except ImportError:
    _requests_available = False


def render():
    """Feedback & suggestions page with built-in form."""
    # Field spacing CSS
    st.markdown("""<style>
    .fb-form [data-testid="stTextArea"],
    .fb-form [data-testid="stTextInput"],
    .fb-form [data-testid="stSelectbox"],
    .fb-form [data-testid="stRadio"] { margin-bottom: 1.2rem !important; }
    </style>""", unsafe_allow_html=True)
    st.markdown(
        "<h2 style='margin-bottom:0.3rem;'>💬 Feedback & Suggestions</h2>"
        "<p style='color:#93b8d8;font-size:0.88rem;margin-bottom:1.2rem;'>"
        "Help us improve MLB Toolbox. Your feedback shapes what we build next.</p>",
        unsafe_allow_html=True,
    )

    _fb_col, _ = st.columns([3, 1])
    with _fb_col:
        _fb_type = st.radio(
            "What kind of feedback?",
            ["💡 Feature Request", "🐛 Bug Report", "📊 Data Issue", "💬 General Feedback"],
            key="fb_page_type", horizontal=True,
        )

        _fb_page = st.selectbox(
            "Which page does this relate to?",
            ["General / Sitewide", "Rankings", "Team Analysis", "Player Analysis",
             "Roster Simulator", "Methodology", "Home Page"],
            key="fb_page_area",
        )

        _fb_text = st.text_area(
            "Your feedback:",
            key="fb_page_text",
            placeholder="Describe the issue, suggestion, or idea in detail...",
            height=200,
        )

        _fb_email = st.text_input(
            "Email (optional — only if you want a response):",
            key="fb_page_email",
            placeholder="your@email.com",
        )

        if st.button("Submit Feedback", key="fb_page_submit", type="primary"):
            if _fb_text.strip():
                _stc.html("""
                <div style="text-align:center;padding:1.5rem 0;">
                  <svg width="60" height="60" viewBox="0 0 60 60">
                    <circle cx="30" cy="30" r="28" fill="none" stroke="#22c55e" stroke-width="3"
                      stroke-dasharray="176" stroke-dashoffset="176"
                      style="animation:circle-draw 0.5s ease-out forwards;" />
                    <path d="M18 30 L26 38 L42 22" fill="none" stroke="#22c55e" stroke-width="3"
                      stroke-linecap="round" stroke-linejoin="round"
                      stroke-dasharray="40" stroke-dashoffset="40"
                      style="animation:check-draw 0.3s ease-out 0.4s forwards;" />
                  </svg>
                  <div style="color:#22c55e;font-size:1.1rem;font-weight:700;margin-top:0.5rem;
                    opacity:0;animation:fadeInText 0.3s ease-out 0.6s forwards;">
                    Thanks for your feedback!</div>
                </div>
                <style>
                @keyframes circle-draw { to { stroke-dashoffset: 0; } }
                @keyframes check-draw { to { stroke-dashoffset: 0; } }
                @keyframes fadeInText { to { opacity: 1; } }
                </style>
                """, height=140)
                # Fire-and-forget: send to Google Sheet with short timeout
                try:
                    if _requests_available:
                        def _send():
                            try:
                                _requests.post(
                                    "https://script.google.com/macros/s/AKfycbxfujsC1uRLp1bD9Bk4JyK6L8Z7ZT4fBgy6vaFRgwGOJc9NYfyX76-9cJ_64cvV6e-NMQ/exec",
                                    json={"type": _fb_type, "page": _fb_page,
                                          "feedback": _fb_text.strip(),
                                          "email": _fb_email.strip() if _fb_email else ""},
                                    timeout=15,
                                )
                            except Exception:
                                pass
                        threading.Thread(target=_send, daemon=True).start()
                except Exception:
                    pass
            else:
                st.warning("Please enter some feedback text before submitting.")

    st.markdown("---")
    st.markdown(
        "<div style='background:#0d1e35;border:1px solid #1e3250;border-radius:8px;"
        "padding:1rem 1.2rem;font-size:0.82rem;color:#7a9ebc;line-height:1.7;'>"
        "<b style='color:#d6e8f8;'>What happens with your feedback?</b><br>"
        "Every submission is reviewed by the MLB Toolbox team. Feature requests are "
        "prioritized based on community demand. Bug reports are triaged and fixed in "
        "order of severity. Data issues are verified against our sources and corrected. "
        "We appreciate your help making this tool better for all baseball fans.</div>",
        unsafe_allow_html=True,
    )
