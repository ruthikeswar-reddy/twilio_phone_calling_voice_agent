"""Generate Voice Pipeline Architecture PPT."""
import pptx
import pptx.oxml.ns as ns
import pptx.enum.shapes
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from lxml import etree

# ── Color Palette ──────────────────────────────────────────────────────────────
DARK_BG      = RGBColor(0x0D, 0x1B, 0x2A)   # deep navy
ACCENT_BLUE  = RGBColor(0x1E, 0x88, 0xE5)   # electric blue
ACCENT_CYAN  = RGBColor(0x00, 0xB8, 0xD4)   # cyan
ACCENT_GREEN = RGBColor(0x43, 0xA0, 0x47)   # green
ACCENT_ORG   = RGBColor(0xFF, 0x8F, 0x00)   # amber/orange
ACCENT_PURP  = RGBColor(0x7B, 0x1F, 0xA2)   # purple
ACCENT_RED   = RGBColor(0xE5, 0x39, 0x35)   # red
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY   = RGBColor(0xB0, 0xBE, 0xC5)
MID_GREY     = RGBColor(0x37, 0x47, 0x4F)
CARD_BG      = RGBColor(0x17, 0x2A, 0x3D)   # card background

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

blank_layout = prs.slide_layouts[6]  # completely blank

# ── Helpers ────────────────────────────────────────────────────────────────────

def add_bg(slide, color=DARK_BG):
    """Fill slide background."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_rect(slide, l, t, w, h, fill_color, line_color=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE_TYPE.AUTO_SHAPE if False else 1,
        l, t, w, h
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width
    else:
        shape.line.fill.background()
    return shape

def add_text_box(slide, text, l, t, w, h, font_size=Pt(12), color=WHITE,
                 bold=False, align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(l, t, w, h)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = font_size
    run.font.color.rgb = color
    run.font.bold  = bold
    return txb

def add_label(slide, text, l, t, w, h, font_size=Pt(11), color=WHITE,
              bold=False, align=PP_ALIGN.CENTER, fill=None, line=None):
    if fill:
        add_rect(slide, l, t, w, h, fill, line, Pt(1.5) if line else Pt(0))
    txb = slide.shapes.add_textbox(l, t, w, h)
    tf  = txb.text_frame
    tf.word_wrap = True
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = font_size
    run.font.color.rgb = color
    run.font.bold  = bold
    return txb

def add_arrow(slide, x1, y1, x2, y2, color=ACCENT_CYAN, width=Pt(2)):
    """Add a connector line with an arrowhead."""
    from pptx.util import Emu
    connector = slide.shapes.add_connector(
        pptx.enum.shapes.MSO_CONNECTOR_TYPE.STRAIGHT, x1, y1, x2, y2
    )
    connector.line.color.rgb = color
    connector.line.width = width
    # arrowhead
    ln = connector.line._ln
    tailEnd = etree.SubElement(ln, ns.qn('a:tailEnd'))
    tailEnd.set('type', 'none')
    headEnd = etree.SubElement(ln, ns.qn('a:headEnd'))
    headEnd.set('type', 'arrow')
    headEnd.set('w', 'med')
    headEnd.set('len', 'med')
    return connector

def section_header(slide, title, subtitle, accent=ACCENT_BLUE):
    """Top bar with slide title."""
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.7), accent)
    add_text_box(slide, title, Inches(0.25), Inches(0.05), Inches(9), Inches(0.6),
                 font_size=Pt(22), bold=True, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text_box(slide, subtitle, Inches(0.25), Inches(0.42), Inches(12), Inches(0.35),
                     font_size=Pt(11), color=LIGHT_GREY, align=PP_ALIGN.LEFT)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ══════════════════════════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(blank_layout)
add_bg(s1)

# gradient bar left
add_rect(s1, 0, 0, Inches(0.3), SLIDE_H, ACCENT_BLUE)
add_rect(s1, Inches(0.3), 0, Inches(0.08), SLIDE_H, ACCENT_CYAN)

# main title
add_text_box(s1, "Voice Agent", Inches(1), Inches(1.6), Inches(11), Inches(1.2),
             font_size=Pt(52), bold=True, align=PP_ALIGN.LEFT)
add_text_box(s1, "Architecture", Inches(1), Inches(2.7), Inches(11), Inches(1.2),
             font_size=Pt(52), bold=True, color=ACCENT_CYAN, align=PP_ALIGN.LEFT)
add_text_box(s1, "Customer Support Voice Pipeline  ·  Real-Time AI Phone Agent",
             Inches(1), Inches(3.8), Inches(11), Inches(0.5),
             font_size=Pt(16), color=LIGHT_GREY, align=PP_ALIGN.LEFT)

# tech pills
pills = [
    ("Twilio Media Streams", ACCENT_BLUE),
    ("Deepgram Nova-3 STT", ACCENT_GREEN),
    ("LangGraph Agent", ACCENT_PURP),
    ("Cartesia Sonic TTS", ACCENT_ORG),
    ("ServiceNow API", ACCENT_RED),
]
px = Inches(1)
for label, col in pills:
    pw = Inches(2.1)
    add_rect(s1, px, Inches(5.0), pw, Inches(0.38), col)
    add_text_box(s1, label, px, Inches(5.02), pw, Inches(0.38),
                 font_size=Pt(11), bold=True, align=PP_ALIGN.CENTER)
    px += pw + Inches(0.15)

add_text_box(s1, "Confidential  ·  2026", Inches(1), Inches(6.9), Inches(5), Inches(0.4),
             font_size=Pt(10), color=LIGHT_GREY)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — SYSTEM OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
s2 = prs.slides.add_slide(blank_layout)
add_bg(s2)
section_header(s2, "System Overview", "End-to-end component view of the voice agent pipeline", ACCENT_BLUE)

# caller box
def box(slide, label, sublabel, l, t, w, h, fill, lcolor=None):
    add_rect(slide, l, t, w, h, fill, lcolor or ACCENT_CYAN, Pt(1.5))
    add_text_box(slide, label, l, t + Inches(0.08), w, Inches(0.35),
                 font_size=Pt(13), bold=True, align=PP_ALIGN.CENTER)
    if sublabel:
        add_text_box(slide, sublabel, l, t + Inches(0.38), w, Inches(0.35),
                     font_size=Pt(9), color=LIGHT_GREY, align=PP_ALIGN.CENTER)

ROW1 = Inches(1.0)
ROW2 = Inches(2.8)
ROW3 = Inches(4.5)
BH   = Inches(0.85)
BW   = Inches(1.8)

boxes = [
    # col_x,    row_y, label,              sub,                      fill
    (Inches(0.3), ROW1, "📞 Caller",       "PSTN Phone",             MID_GREY),
    (Inches(2.4), ROW1, "Twilio",          "Media Streams\nWebSocket", ACCENT_BLUE),
    (Inches(4.5), ROW1, "FastAPI Server",  "main.py\n/media-stream", ACCENT_BLUE),
    (Inches(6.6), ROW1, "VoicePipeline",   "voice_pipeline.py\nOrchestrator", ACCENT_CYAN),
    (Inches(0.3), ROW2, "Deepgram STT",    "Nova-3\nStreaming ASR",  ACCENT_GREEN),
    (Inches(2.4), ROW2, "LangGraph Agent", "csa_groq.py\nState Machine", ACCENT_PURP),
    (Inches(4.5), ROW2, "Cartesia TTS",    "Sonic English\nStreaming", ACCENT_ORG),
    (Inches(6.6), ROW2, "Audio Utils",     "PCM ↔ μ-law\nCodec",    MID_GREY),
    (Inches(2.4), ROW3, "Groq / OpenAI",   "LLM Provider\n(Streaming)", ACCENT_PURP),
    (Inches(4.5), ROW3, "ServiceNow",      "Incident API\nTicket Creation", ACCENT_RED),
]

for (lx, ty, lbl, sub, fill) in boxes:
    box(s2, lbl, sub, lx, ty, BW, BH, fill)

# arrows (simplified horizontal / vertical)
arrows_s2 = [
    # (x1, y1, x2, y2)
    (Inches(2.1), ROW1+BH/2, Inches(2.4), ROW1+BH/2),  # caller→twilio
    (Inches(4.2), ROW1+BH/2, Inches(4.5), ROW1+BH/2),  # twilio→fastapi
    (Inches(6.3), ROW1+BH/2, Inches(6.6), ROW1+BH/2),  # fastapi→pipeline
    # pipeline → stt
    (Inches(1.3), ROW1+BH,   Inches(1.3), ROW2),
    # pipeline → tts
    (Inches(5.4), ROW1+BH,   Inches(5.4), ROW2),
    # stt → agent
    (Inches(2.1), ROW2+BH/2, Inches(2.4), ROW2+BH/2),
    # agent → tts
    (Inches(4.2), ROW2+BH/2, Inches(4.5), ROW2+BH/2),
    # tts → audio utils
    (Inches(6.3), ROW2+BH/2, Inches(6.6), ROW2+BH/2),
    # agent → groq
    (Inches(3.3), ROW2+BH,   Inches(3.3), ROW3),
    # agent → servicenow
    (Inches(5.4), ROW2+BH,   Inches(5.4), ROW3),
]
for (x1,y1,x2,y2) in arrows_s2:
    add_arrow(s2, x1, y1, x2, y2, ACCENT_CYAN, Pt(1.5))

# legend on right
add_rect(s2, Inches(8.7), Inches(1.0), Inches(4.3), Inches(5.8), CARD_BG, ACCENT_BLUE, Pt(1))
add_text_box(s2, "Component Legend", Inches(8.9), Inches(1.1), Inches(4.0), Inches(0.35),
             font_size=Pt(13), bold=True, align=PP_ALIGN.LEFT)
legend_items = [
    (ACCENT_BLUE,  "Twilio / Network Layer"),
    (ACCENT_GREEN, "Speech-to-Text (STT)"),
    (ACCENT_PURP,  "AI Agent / LLM"),
    (ACCENT_ORG,   "Text-to-Speech (TTS)"),
    (ACCENT_RED,   "External APIs"),
    (MID_GREY,     "Infrastructure / Utilities"),
]
for i, (col, lbl) in enumerate(legend_items):
    ty = Inches(1.6) + i * Inches(0.7)
    add_rect(s2, Inches(8.9), ty, Inches(0.3), Inches(0.28), col)
    add_text_box(s2, lbl, Inches(9.3), ty - Inches(0.03), Inches(3.5), Inches(0.38),
                 font_size=Pt(12), color=WHITE)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — DATA FLOW (6 PHASES)
# ══════════════════════════════════════════════════════════════════════════════
s3 = prs.slides.add_slide(blank_layout)
add_bg(s3)
section_header(s3, "Data Flow — 6-Phase Audio Pipeline",
               "Real-time path from caller speech to synthesized voice response", ACCENT_CYAN)

phases = [
    ("1", "INCOMING\nAUDIO",    "Twilio WebSocket\nμ-law 8kHz audio",       ACCENT_BLUE),
    ("2", "SPEECH-TO-\nTEXT",   "Deepgram Nova-3\nVAD + ASR streaming",     ACCENT_GREEN),
    ("3", "AGENT\nPROCESSING",  "LangGraph\nState Machine",                  ACCENT_PURP),
    ("4", "TEXT-TO-\nSPEECH",   "Cartesia Sonic\nPCM streaming",            ACCENT_ORG),
    ("5", "OUTBOUND\nAUDIO",     "PCM→μ-law encode\nBack to Twilio",         ACCENT_BLUE),
    ("6", "BARGE-IN\nDETECTION","VAD interruption\n2-step arm+fire",         ACCENT_RED),
]

PH = Inches(1.15)
PW = Inches(1.9)
GAP = Inches(0.2)
TOTAL = len(phases) * PW + (len(phases)-1) * GAP
START_X = (SLIDE_W - TOTAL) / 2

for i, (num, title, detail, color) in enumerate(phases):
    lx = START_X + i*(PW+GAP)
    ty = Inches(1.0)
    # number circle header
    add_rect(s3, lx, ty, PW, Inches(0.42), color)
    add_text_box(s3, f"PHASE {num}", lx, ty+Inches(0.04), PW, Inches(0.38),
                 font_size=Pt(10), bold=True, align=PP_ALIGN.CENTER)
    # card
    add_rect(s3, lx, ty+Inches(0.42), PW, PH, CARD_BG, color, Pt(1.2))
    add_text_box(s3, title, lx, ty+Inches(0.52), PW, Inches(0.55),
                 font_size=Pt(13), bold=True, align=PP_ALIGN.CENTER)
    add_text_box(s3, detail, lx, ty+Inches(1.0), PW, Inches(0.65),
                 font_size=Pt(9.5), color=LIGHT_GREY, align=PP_ALIGN.CENTER)
    # arrow between phases (not after last)
    if i < len(phases)-1:
        ax = lx + PW
        ay = ty + Inches(0.42) + PH/2
        add_arrow(s3, ax, ay, ax+GAP, ay, ACCENT_CYAN, Pt(2))

# Latency bar below
add_rect(s3, Inches(0.3), Inches(3.0), SLIDE_W - Inches(0.6), Inches(0.05), ACCENT_CYAN)
add_text_box(s3, "Target: < 500ms Time-To-First-Word (TTFW)",
             Inches(0.3), Inches(3.1), SLIDE_W - Inches(0.6), Inches(0.35),
             font_size=Pt(12), color=ACCENT_CYAN, bold=True, align=PP_ALIGN.CENTER)

# Detail cards below
detail_cards = [
    ("Twilio Protocol", "JSON events over WebSocket\n• connected → start → media → stop\n• Base64 μ-law 8kHz chunks\n• 'clear' for barge-in interrupt", ACCENT_BLUE),
    ("Deepgram VAD", "Voice Activity Detection\n• speech_start → arm barge-in\n• is_final → accumulate buffer\n• speech_final → launch agent", ACCENT_GREEN),
    ("LangGraph Nodes", "classify → extract → ask\n→ confirm → handle_confirm\n→ create (ServiceNow ticket)", ACCENT_PURP),
    ("Cartesia TTS", "Min 100-char buffer\nSentence splitting (120 char)\nPCM 16-bit LE at 8kHz\n~80–120ms first chunk", ACCENT_ORG),
]
DW = Inches(3.1)
DH = Inches(2.65)
dx = Inches(0.2)
dy = Inches(3.6)
for i, (title, body, col) in enumerate(detail_cards):
    lx = dx + i*(DW + Inches(0.18))
    add_rect(s3, lx, dy, DW, Inches(0.38), col)
    add_text_box(s3, title, lx, dy+Inches(0.04), DW, Inches(0.35),
                 font_size=Pt(11), bold=True, align=PP_ALIGN.CENTER)
    add_rect(s3, lx, dy+Inches(0.38), DW, DH-Inches(0.38), CARD_BG, col, Pt(1))
    add_text_box(s3, body, lx+Inches(0.1), dy+Inches(0.45), DW-Inches(0.2), DH-Inches(0.55),
                 font_size=Pt(9.5), color=LIGHT_GREY)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — VOICE PIPELINE (3 ASYNC LOOPS)
# ══════════════════════════════════════════════════════════════════════════════
s4 = prs.slides.add_slide(blank_layout)
add_bg(s4)
section_header(s4, "VoicePipeline — 3 Concurrent Async Loops",
               "voice_pipeline.py · asyncio TaskGroup orchestrator", ACCENT_PURP)

loops = [
    ("LOOP 1", "_receive_twilio_loop()", [
        "Read Twilio WebSocket",
        "Decode base64 μ-law audio",
        "Forward chunks to Deepgram STT",
        "Handle connect / start / stop events",
    ], ACCENT_BLUE),
    ("LOOP 2", "_handle_stt_events()", [
        "Process speech_start → arm barge-in",
        "Process interim → fire barge-in",
        "Process final → normalize transcript",
        "Launch/cancel agent task",
    ], ACCENT_GREEN),
    ("LOOP 3", "_tts_loop() + _send_audio_loop()", [
        "Buffer agent tokens (min 100 chars)",
        "Split on sentences / newlines",
        "Stream to Cartesia TTS API",
        "PCM → μ-law → base64 → Twilio",
    ], ACCENT_ORG),
]

LW = Inches(3.8)
LH = Inches(4.5)
LY = Inches(0.85)
for i, (badge, title, items, col) in enumerate(loops):
    lx = Inches(0.25) + i*(LW + Inches(0.2))
    # header
    add_rect(s4, lx, LY, LW, Inches(0.45), col)
    add_text_box(s4, badge, lx, LY+Inches(0.04), LW, Inches(0.38),
                 font_size=Pt(11), bold=True, align=PP_ALIGN.CENTER)
    # body
    add_rect(s4, lx, LY+Inches(0.45), LW, LH-Inches(0.45), CARD_BG, col, Pt(1.5))
    add_text_box(s4, title, lx+Inches(0.1), LY+Inches(0.52), LW-Inches(0.2), Inches(0.4),
                 font_size=Pt(11.5), bold=True, color=col)
    for j, item in enumerate(items):
        add_text_box(s4, f"▶  {item}",
                     lx+Inches(0.15), LY+Inches(1.05)+j*Inches(0.72),
                     LW-Inches(0.3), Inches(0.65),
                     font_size=Pt(10.5), color=WHITE)

# queue indicators
add_rect(s4, Inches(4.05), Inches(3.2), Inches(0.5), Inches(0.35), ACCENT_CYAN)
add_text_box(s4, "Queue", Inches(4.05), Inches(3.22), Inches(0.5), Inches(0.32),
             font_size=Pt(7), bold=True, align=PP_ALIGN.CENTER)
add_rect(s4, Inches(8.05), Inches(3.2), Inches(0.5), Inches(0.35), ACCENT_CYAN)
add_text_box(s4, "Queue", Inches(8.05), Inches(3.22), Inches(0.5), Inches(0.32),
             font_size=Pt(7), bold=True, align=PP_ALIGN.CENTER)

# barge-in note
add_rect(s4, Inches(0.25), Inches(5.5), Inches(12.3), Inches(1.7), CARD_BG, ACCENT_RED, Pt(1.5))
add_text_box(s4, "Barge-In (Interruption) Flow", Inches(0.4), Inches(5.55), Inches(12), Inches(0.38),
             font_size=Pt(13), bold=True, color=ACCENT_RED)
barge_steps = [
    "1. speech_start VAD event fires",
    "2. Check elapsed TTS time ≥ 700ms grace period",
    "3. Set _barge_in_armed = True",
    "4. Interim/final transcript confirms real speech",
    "5. Cancel _agent_task · drain queues · send 'clear' to Twilio",
]
for i, step in enumerate(barge_steps):
    add_text_box(s4, step, Inches(0.4) + i*Inches(2.46), Inches(5.97), Inches(2.4), Inches(0.5),
                 font_size=Pt(9), color=LIGHT_GREY)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — STT (DEEPGRAM)
# ══════════════════════════════════════════════════════════════════════════════
s5 = prs.slides.add_slide(blank_layout)
add_bg(s5)
section_header(s5, "Speech-to-Text — Deepgram Nova-3",
               "stt/deepgram_stt.py  ·  Streaming ASR over WebSocket", ACCENT_GREEN)

# Config table
add_rect(s5, Inches(0.3), Inches(0.85), Inches(5.5), Inches(5.9), CARD_BG, ACCENT_GREEN, Pt(1.2))
add_text_box(s5, "Configuration", Inches(0.4), Inches(0.95), Inches(5.3), Inches(0.38),
             font_size=Pt(14), bold=True, color=ACCENT_GREEN)
cfg_rows = [
    ("Model",           "nova-3"),
    ("Input Format",    "μ-law 8kHz (Twilio native)"),
    ("Protocol",        "WebSocket streaming"),
    ("VAD",             "Enabled (vad_events=True)"),
    ("Endpointing",     "200ms silence threshold"),
    ("Utterance End",   "1000ms timeout"),
    ("Smart Format",    "Enabled (punctuation)"),
    ("Language",        "en-US"),
]
for i, (k, v) in enumerate(cfg_rows):
    ty = Inches(1.45) + i*Inches(0.57)
    bg = DARK_BG if i % 2 == 0 else CARD_BG
    add_rect(s5, Inches(0.35), ty, Inches(5.4), Inches(0.52), bg)
    add_text_box(s5, k, Inches(0.45), ty+Inches(0.08), Inches(2.0), Inches(0.38),
                 font_size=Pt(10.5), bold=True, color=ACCENT_GREEN)
    add_text_box(s5, v, Inches(2.5), ty+Inches(0.08), Inches(3.2), Inches(0.38),
                 font_size=Pt(10.5), color=LIGHT_GREY)

# Events diagram
add_rect(s5, Inches(6.0), Inches(0.85), Inches(7.0), Inches(5.9), CARD_BG, ACCENT_GREEN, Pt(1.2))
add_text_box(s5, "Events & Processing Logic", Inches(6.1), Inches(0.95), Inches(6.8), Inches(0.38),
             font_size=Pt(14), bold=True, color=ACCENT_GREEN)

events = [
    ("speech_start",  ACCENT_GREEN,  "VAD detects voice activity",
     "→ Arm barge-in (after 700ms grace)"),
    ("interim",       ACCENT_CYAN,   "Partial / committed chunk (is_final=True)",
     "→ Fire barge-in if armed"),
    ("final",         ACCENT_ORG,    "Utterance complete (speech_final=True)",
     "→ Normalize transcript → launch agent"),
    ("utterance_end", ACCENT_PURP,   "1000ms silence detected",
     "→ Flush accumulated is_final buffer"),
]
for i, (evt, col, desc, action) in enumerate(events):
    ty = Inches(1.45) + i*Inches(1.2)
    add_rect(s5, Inches(6.1), ty, Inches(6.8), Inches(1.1), DARK_BG, col, Pt(1))
    add_rect(s5, Inches(6.1), ty, Inches(2.1), Inches(1.1), col)
    add_text_box(s5, evt, Inches(6.15), ty+Inches(0.32), Inches(2.0), Inches(0.45),
                 font_size=Pt(11), bold=True, align=PP_ALIGN.CENTER)
    add_text_box(s5, desc, Inches(8.3), ty+Inches(0.07), Inches(4.5), Inches(0.45),
                 font_size=Pt(10), color=WHITE)
    add_text_box(s5, action, Inches(8.3), ty+Inches(0.55), Inches(4.5), Inches(0.45),
                 font_size=Pt(9.5), color=col)

# Transcript normalization note
add_rect(s5, Inches(0.3), Inches(6.88), Inches(12.7), Inches(0.52), CARD_BG, ACCENT_CYAN, Pt(1))
add_text_box(s5,
    '  Transcript Normalization:  "dot com" → ".com"  ·  "at the rate" → "@"  ·  '
    '"one two three" → "123"  ·  "E M one" → "EM1"',
    Inches(0.4), Inches(6.91), Inches(12.5), Inches(0.45),
    font_size=Pt(10), color=ACCENT_CYAN, bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — LANGGRAPH AGENT
# ══════════════════════════════════════════════════════════════════════════════
s6 = prs.slides.add_slide(blank_layout)
add_bg(s6)
section_header(s6, "LangGraph Agent — ServiceNow Ticket Creation",
               "agent/csa_groq.py  ·  StateGraph with 7 nodes and conditional routing", ACCENT_PURP)

# State schema box
add_rect(s6, Inches(0.2), Inches(0.85), Inches(3.8), Inches(5.9), CARD_BG, ACCENT_PURP, Pt(1.2))
add_text_box(s6, "TicketState Schema", Inches(0.3), Inches(0.95), Inches(3.6), Inches(0.38),
             font_size=Pt(13), bold=True, color=ACCENT_PURP)
state_fields = [
    ("messages",             "List[BaseMessage]  (history)"),
    ("slots",                "{employee_id, email, location,\n short_desc, detail_desc, category, priority}"),
    ("pending_field",        "Optional[str]  (next to collect)"),
    ("awaiting_confirmation","bool  (summary shown to user)"),
    ("confirmed",            "bool  (user agreed)"),
    ("field_to_update",      "Optional[str]  (change request)"),
]
for i, (k, v) in enumerate(state_fields):
    ty = Inches(1.45) + i*Inches(0.72)
    add_rect(s6, Inches(0.25), ty, Inches(3.7), Inches(0.68),
             DARK_BG if i%2==0 else CARD_BG)
    add_text_box(s6, k, Inches(0.32), ty+Inches(0.04), Inches(1.8), Inches(0.32),
                 font_size=Pt(9.5), bold=True, color=ACCENT_PURP)
    add_text_box(s6, v, Inches(0.32), ty+Inches(0.3), Inches(3.6), Inches(0.38),
                 font_size=Pt(8.5), color=LIGHT_GREY)

# Graph nodes
node_data = [
    # (label, x, y, color)
    ("classify",       Inches(5.3),  Inches(1.2),  ACCENT_PURP),
    ("greeting",       Inches(3.5),  Inches(2.4),  ACCENT_BLUE),
    ("unrelated",      Inches(7.2),  Inches(2.4),  ACCENT_RED),
    ("extract",        Inches(5.3),  Inches(2.4),  ACCENT_PURP),
    ("ask",            Inches(4.2),  Inches(3.6),  ACCENT_ORG),
    ("confirm",        Inches(6.4),  Inches(3.6),  ACCENT_ORG),
    ("handle_confirm", Inches(5.3),  Inches(4.7),  ACCENT_CYAN),
    ("create",         Inches(5.3),  Inches(5.7),  ACCENT_GREEN),
]
NW, NH = Inches(1.7), Inches(0.52)

for (lbl, nx, ny, col) in node_data:
    add_rect(s6, nx, ny, NW, NH, col, WHITE, Pt(1))
    add_text_box(s6, lbl, nx, ny+Inches(0.1), NW, NH,
                 font_size=Pt(11), bold=True, align=PP_ALIGN.CENTER)

# Edges (from center of each node)
def nc(nx, ny): return (nx + NW/2, ny + NH/2)

classify_c   = nc(Inches(5.3),  Inches(1.2))
greeting_c   = nc(Inches(3.5),  Inches(2.4))
unrelated_c  = nc(Inches(7.2),  Inches(2.4))
extract_c    = nc(Inches(5.3),  Inches(2.4))
ask_c        = nc(Inches(4.2),  Inches(3.6))
confirm_c    = nc(Inches(6.4),  Inches(3.6))
h_confirm_c  = nc(Inches(5.3),  Inches(4.7))
create_c     = nc(Inches(5.3),  Inches(5.7))

edges = [
    (classify_c, greeting_c),
    (classify_c, unrelated_c),
    (classify_c, extract_c),
    (extract_c,  ask_c),
    (extract_c,  confirm_c),
    (ask_c,      h_confirm_c),
    (confirm_c,  h_confirm_c),
    (h_confirm_c,create_c),
    (greeting_c, create_c),
    (unrelated_c,create_c),
]
for (x1,y1),(x2,y2) in edges:
    add_arrow(s6, x1, y1, x2, y2, ACCENT_CYAN, Pt(1.5))

# Node descriptions on right
add_rect(s6, Inches(9.2), Inches(0.85), Inches(3.9), Inches(5.9), CARD_BG, ACCENT_PURP, Pt(1))
add_text_box(s6, "Node Descriptions", Inches(9.3), Inches(0.95), Inches(3.7), Inches(0.38),
             font_size=Pt(13), bold=True, color=ACCENT_PURP)
node_descs = [
    ("classify",       "Routes: greeting / ticket / unrelated"),
    ("greeting",       "Welcome response for salutations"),
    ("unrelated",      "Redirect to ticket creation"),
    ("extract",        "LLM extracts ticket fields from speech"),
    ("ask",            "Voice-friendly question for missing field"),
    ("confirm",        "Summarize ticket, ask user to confirm"),
    ("handle_confirm", "Route: confirm / change / unclear"),
    ("create",         "HTTP POST to ServiceNow incident API"),
]
for i, (n, d) in enumerate(node_descs):
    ty = Inches(1.45) + i*Inches(0.6)
    add_text_box(s6, f"  {n}", Inches(9.3), ty, Inches(1.7), Inches(0.5),
                 font_size=Pt(9.5), bold=True, color=ACCENT_PURP)
    add_text_box(s6, d, Inches(11.05), ty, Inches(1.9), Inches(0.5),
                 font_size=Pt(9), color=LIGHT_GREY)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — TTS & AUDIO
# ══════════════════════════════════════════════════════════════════════════════
s7 = prs.slides.add_slide(blank_layout)
add_bg(s7)
section_header(s7, "Text-to-Speech & Audio Pipeline",
               "tts/cartesia_tts.py  ·  audio_utils.py  ·  Cartesia Sonic streaming TTS", ACCENT_ORG)

# Cartesia spec
add_rect(s7, Inches(0.2), Inches(0.85), Inches(5.8), Inches(3.3), CARD_BG, ACCENT_ORG, Pt(1.2))
add_text_box(s7, "Cartesia Sonic — Specs", Inches(0.32), Inches(0.95), Inches(5.5), Inches(0.38),
             font_size=Pt(14), bold=True, color=ACCENT_ORG)
cartesia_cfg = [
    ("Model",           "sonic-english"),
    ("Output Format",   "PCM 16-bit LE at 8 kHz"),
    ("API",             "HTTPS POST streaming"),
    ("Container",       "raw (no wrapper)"),
    ("Encoding",        "pcm_s16le"),
    ("Latency",         "~80–120ms to first chunk"),
]
for i, (k, v) in enumerate(cartesia_cfg):
    ty = Inches(1.45) + i*Inches(0.43)
    bg = DARK_BG if i%2==0 else CARD_BG
    add_rect(s7, Inches(0.25), ty, Inches(5.7), Inches(0.4), bg)
    add_text_box(s7, k, Inches(0.35), ty+Inches(0.05), Inches(2.0), Inches(0.32),
                 font_size=Pt(10), bold=True, color=ACCENT_ORG)
    add_text_box(s7, v, Inches(2.4), ty+Inches(0.05), Inches(3.5), Inches(0.32),
                 font_size=Pt(10), color=LIGHT_GREY)

# Audio pipeline flow
add_rect(s7, Inches(0.2), Inches(4.3), Inches(5.8), Inches(2.5), CARD_BG, ACCENT_ORG, Pt(1.2))
add_text_box(s7, "Audio Codec Utilities", Inches(0.32), Inches(4.4), Inches(5.5), Inches(0.38),
             font_size=Pt(14), bold=True, color=ACCENT_ORG)
audio_fns = [
    "pcm_to_ulaw(pcm_bytes)    — PCM 16-bit LE → μ-law 8-bit",
    "ulaw_to_pcm(ulaw_bytes)   — μ-law 8-bit → PCM 16-bit LE",
    "resample_pcm(pcm, fr, to) — Resample between sample rates",
    "normalize_audio_level()   — Normalize RMS volume",
]
for i, fn in enumerate(audio_fns):
    add_text_box(s7, fn, Inches(0.35), Inches(4.85)+i*Inches(0.45), Inches(5.6), Inches(0.4),
                 font_size=Pt(9.5), color=LIGHT_GREY)

# TTS flow diagram
add_rect(s7, Inches(6.2), Inches(0.85), Inches(6.9), Inches(5.95), CARD_BG, ACCENT_ORG, Pt(1.2))
add_text_box(s7, "TTS Buffering & Streaming Flow", Inches(6.35), Inches(0.95), Inches(6.6), Inches(0.38),
             font_size=Pt(14), bold=True, color=ACCENT_ORG)

tts_steps = [
    ("Agent Text Tokens",     "Word/chunk tokens from LangGraph", ACCENT_PURP),
    ("Token Buffer",          "Accumulate until ≥ 100 chars\nor 50ms timeout elapses",  ACCENT_ORG),
    ("Sentence Splitter",     "Split at '.', '?', '!', newlines\nMax 120 chars per segment", ACCENT_CYAN),
    ("Cartesia TTS API",      "HTTPS POST streaming request\nReturns raw PCM chunks",  ACCENT_ORG),
    ("PCM → μ-law Encoder",   "audioop.lin2ulaw conversion\nRun in executor (non-blocking)", ACCENT_GREEN),
    ("Twilio WebSocket",      "Base64-encode → JSON media event\nSend to caller", ACCENT_BLUE),
]
for i, (title, detail, col) in enumerate(tts_steps):
    ty = Inches(1.45) + i*Inches(0.78)
    add_rect(s7, Inches(6.3), ty, Inches(6.7), Inches(0.72), DARK_BG, col, Pt(1))
    add_rect(s7, Inches(6.3), ty, Inches(0.18), Inches(0.72), col)
    add_text_box(s7, title, Inches(6.6), ty+Inches(0.04), Inches(3.5), Inches(0.32),
                 font_size=Pt(10.5), bold=True, color=col)
    add_text_box(s7, detail, Inches(6.6), ty+Inches(0.36), Inches(6.2), Inches(0.32),
                 font_size=Pt(9), color=LIGHT_GREY)
    if i < len(tts_steps)-1:
        ax = Inches(6.3) + Inches(3.35)
        ay = ty + Inches(0.72)
        add_arrow(s7, ax, ay, ax, ay+Inches(0.06), ACCENT_CYAN, Pt(1.5))

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — EXTERNAL SERVICES & LATENCY
# ══════════════════════════════════════════════════════════════════════════════
s8 = prs.slides.add_slide(blank_layout)
add_bg(s8)
section_header(s8, "External Services & Latency Budget",
               "API integrations and time-to-first-word breakdown", ACCENT_RED)

# External services table
add_rect(s8, Inches(0.2), Inches(0.85), Inches(7.8), Inches(4.4), CARD_BG, ACCENT_RED, Pt(1.2))
add_text_box(s8, "External Services", Inches(0.35), Inches(0.95), Inches(7.5), Inches(0.38),
             font_size=Pt(14), bold=True, color=ACCENT_RED)

# Header row
hdr_cols = ["Service", "Purpose", "Protocol", "Key Detail"]
hdr_widths = [Inches(1.6), Inches(2.3), Inches(1.8), Inches(2.0)]
hx = Inches(0.25)
for col, w in zip(hdr_cols, hdr_widths):
    add_rect(s8, hx, Inches(1.42), w - Inches(0.02), Inches(0.38), ACCENT_RED)
    add_text_box(s8, col, hx, Inches(1.46), w, Inches(0.32),
                 font_size=Pt(10), bold=True, align=PP_ALIGN.CENTER)
    hx += w

services = [
    ("Twilio",        "Phone routing",      "WebSocket JSON",    "Media Streams protocol",    ACCENT_BLUE),
    ("Deepgram",      "Speech-to-Text",     "WebSocket stream",  "Nova-3 model, VAD enabled", ACCENT_GREEN),
    ("Groq / OpenAI", "LLM inference",      "HTTPS streaming",   "LangChain integration",     ACCENT_PURP),
    ("Cartesia",      "Text-to-Speech",     "HTTPS streaming",   "sonic-english, PCM output", ACCENT_ORG),
    ("ServiceNow",    "Ticket creation",    "HTTPS REST",        "Incident table API",        ACCENT_RED),
]
for i, (svc, purpose, proto, detail, col) in enumerate(services):
    ty = Inches(1.82) + i*Inches(0.5)
    bg = DARK_BG if i%2==0 else CARD_BG
    row_data = [svc, purpose, proto, detail]
    hx = Inches(0.25)
    for j, (text, w) in enumerate(zip(row_data, hdr_widths)):
        add_rect(s8, hx, ty, w - Inches(0.02), Inches(0.46), bg)
        txt_color = col if j == 0 else (WHITE if j < 3 else LIGHT_GREY)
        add_text_box(s8, text, hx+Inches(0.05), ty+Inches(0.09), w-Inches(0.1), Inches(0.3),
                     font_size=Pt(10), color=txt_color, bold=(j==0))
        hx += w

# Latency budget
add_rect(s8, Inches(0.2), Inches(5.4), Inches(7.8), Inches(1.85), CARD_BG, ACCENT_CYAN, Pt(1.2))
add_text_box(s8, "Latency Budget (Time-To-First-Word)", Inches(0.35), Inches(5.5), Inches(7.5), Inches(0.38),
             font_size=Pt(13), bold=True, color=ACCENT_CYAN)
lat_items = [
    ("Deepgram STT (is_final)",     "80–150ms",  0.40),
    ("LLM Time-To-First-Token",     "50–200ms",  0.50),
    ("Cartesia TTS first chunk",    "80–120ms",  0.30),
    ("μ-law encode + network",      "10–30ms",   0.10),
]
bar_start = Inches(0.25)
bar_w_total = Inches(7.6)
bar_y = Inches(6.1)
bar_h = Inches(0.45)
colors_lat = [ACCENT_GREEN, ACCENT_PURP, ACCENT_ORG, ACCENT_BLUE]
prev = bar_start
for i, ((label, ms, ratio), col) in enumerate(zip(lat_items, colors_lat)):
    seg_w = bar_w_total * ratio
    add_rect(s8, prev, bar_y, seg_w, bar_h, col)
    add_text_box(s8, f"{label}\n{ms}", prev, bar_y - Inches(0.55), seg_w, Inches(0.5),
                 font_size=Pt(7.5), color=col, align=PP_ALIGN.CENTER)
    prev += seg_w
add_text_box(s8, "Total: 220–500ms  (target < 500ms TTFW)",
             Inches(0.25), bar_y + bar_h + Inches(0.08), Inches(7.6), Inches(0.35),
             font_size=Pt(11), bold=True, color=ACCENT_CYAN, align=PP_ALIGN.CENTER)

# Design patterns box
add_rect(s8, Inches(8.2), Inches(0.85), Inches(4.9), Inches(6.4), CARD_BG, ACCENT_CYAN, Pt(1.2))
add_text_box(s8, "Key Design Patterns", Inches(8.35), Inches(0.95), Inches(4.6), Inches(0.38),
             font_size=Pt(14), bold=True, color=ACCENT_CYAN)
patterns = [
    ("Async TaskGroup",       "3 independent loops via asyncio\nTaskGroup; any exception stops all"),
    ("Queue-Based Decoupling","STT → Agent → TTS via asyncio.Queue\nProducer/consumer pattern"),
    ("Streaming Everywhere",  "WebSockets for STT/Twilio\nHTTPS streaming for TTS/LLM"),
    ("Speculative Execution", "Agent starts on interim STT result\nFinal transcript triggers real run"),
    ("State Machine",         "LangGraph ticket collection\nMulti-node graph with routing"),
    ("Per-Call Isolation",    "Each Twilio call gets fresh\nVoicePipeline + Agent instance"),
]
for i, (title, body) in enumerate(patterns):
    ty = Inches(1.45) + i*Inches(0.88)
    add_rect(s8, Inches(8.25), ty, Inches(4.8), Inches(0.82), DARK_BG, ACCENT_CYAN, Pt(0.5))
    add_text_box(s8, title, Inches(8.35), ty+Inches(0.04), Inches(4.6), Inches(0.32),
                 font_size=Pt(10.5), bold=True, color=ACCENT_CYAN)
    add_text_box(s8, body, Inches(8.35), ty+Inches(0.35), Inches(4.6), Inches(0.44),
                 font_size=Pt(9), color=LIGHT_GREY)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
OUT = "/Users/ruthikeswar.t/Desktop/EM/VOICEAGENTS/voice-agent-updated/VoiceAgent_Architecture.pptx"
prs.save(OUT)
print(f"Saved: {OUT}")
