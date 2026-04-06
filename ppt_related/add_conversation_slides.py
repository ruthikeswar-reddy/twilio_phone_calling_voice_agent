"""Append 3 conversation slides to VoiceAgent_Architecture.pptx."""
import pptx
import pptx.oxml.ns as ns
import pptx.enum.shapes
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from lxml import etree

# ── Color Palette (same as main deck) ─────────────────────────────────────────
DARK_BG      = RGBColor(0x0D, 0x1B, 0x2A)
ACCENT_BLUE  = RGBColor(0x1E, 0x88, 0xE5)
ACCENT_CYAN  = RGBColor(0x00, 0xB8, 0xD4)
ACCENT_GREEN = RGBColor(0x43, 0xA0, 0x47)
ACCENT_ORG   = RGBColor(0xFF, 0x8F, 0x00)
ACCENT_PURP  = RGBColor(0x7B, 0x1F, 0xA2)
ACCENT_RED   = RGBColor(0xE5, 0x39, 0x35)
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY   = RGBColor(0xB0, 0xBE, 0xC5)
MID_GREY     = RGBColor(0x37, 0x47, 0x4F)
CARD_BG      = RGBColor(0x17, 0x2A, 0x3D)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

FILE = "/Users/ruthikeswar.t/Desktop/EM/VOICEAGENTS/voice-agent-updated/VoiceAgent_Architecture.pptx"
prs = Presentation(FILE)
blank_layout = prs.slide_layouts[6]

# ── Helpers ───────────────────────────────────────────────────────────────────

def add_bg(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = DARK_BG

def add_rect(slide, l, t, w, h, fill_color, line_color=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(1, l, t, w, h)
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

def section_header(slide, title, subtitle, badge_text, badge_color):
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.68), ACCENT_BLUE)
    # badge pill on left
    add_rect(slide, Inches(0.2), Inches(0.12), Inches(1.5), Inches(0.44), badge_color)
    add_text_box(slide, badge_text, Inches(0.2), Inches(0.14), Inches(1.5), Inches(0.42),
                 font_size=Pt(10), bold=True, align=PP_ALIGN.CENTER)
    add_text_box(slide, title, Inches(1.85), Inches(0.04), Inches(9), Inches(0.4),
                 font_size=Pt(20), bold=True)
    add_text_box(slide, subtitle, Inches(1.85), Inches(0.38), Inches(11), Inches(0.3),
                 font_size=Pt(10), color=LIGHT_GREY)

# ── Bubble renderer ───────────────────────────────────────────────────────────
USER_COL  = RGBColor(0x1E, 0x88, 0xE5)   # blue  – right side
AGENT_COL = RGBColor(0x1B, 0x3A, 0x52)   # dark teal – left side
USER_LABEL_COL  = ACCENT_CYAN
AGENT_LABEL_COL = ACCENT_GREEN

def draw_bubbles(slide, turns, start_y, col_left_x, col_right_x, bubble_w):
    """
    turns: list of ("user"|"agent", text)
    Returns the y position after last bubble.
    """
    y = start_y
    PAD_V = Inches(0.1)
    LABEL_H = Inches(0.28)
    BUBBLE_PAD_H = Inches(0.15)
    LINE_H = Pt(11)

    for speaker, text in turns:
        is_user = (speaker == "user")

        # Estimate height: roughly 1 line per ~60 chars
        lines = max(1, len(text) // 58 + text.count("\n"))
        bubble_h = Inches(0.22) + lines * Inches(0.24)

        if is_user:
            lx = col_right_x
            label = "User"
            label_color = USER_LABEL_COL
            bubble_fill = USER_COL
        else:
            lx = col_left_x
            label = "Customer Support Voice Agent"
            label_color = AGENT_LABEL_COL
            bubble_fill = AGENT_COL

        # label
        add_text_box(slide, label, lx, y, bubble_w, LABEL_H,
                     font_size=Pt(9), color=label_color, bold=True)
        y += LABEL_H

        # bubble
        add_rect(slide, lx, y, bubble_w, bubble_h, bubble_fill,
                 label_color, Pt(0.75))
        add_text_box(slide, text,
                     lx + BUBBLE_PAD_H, y + Inches(0.07),
                     bubble_w - 2*BUBBLE_PAD_H, bubble_h - Inches(0.1),
                     font_size=Pt(10), color=WHITE, wrap=True)
        y += bubble_h + PAD_V

    return y


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE — CASE 1: HAPPY PATH (split across 2 columns)
# ══════════════════════════════════════════════════════════════════════════════
s_c1 = prs.slides.add_slide(blank_layout)
add_bg(s_c1)
section_header(s_c1,
               "Case 1 — Happy Path  ·  End-to-End Ticket Creation",
               "User provides all details correctly; ticket created without any corrections",
               "CASE 1", ACCENT_GREEN)

# Left column
COL_L = Inches(0.25)
COL_R = Inches(6.85)
BW    = Inches(6.25)
START_Y = Inches(0.8)

turns_left = [
    ("user",  "Hello. How can you help me today?"),
    ("agent", "Hi! I can take care of creating your ServiceNow ticket. Could you describe the issue you're facing?"),
    ("user",  "I'm not able to log in into my VDI."),
    ("agent", "Can you provide more details about the issue you're experiencing?"),
    ("user",  "After providing the credentials it is taking too much time to log in, and I am not able to see anything."),
    ("agent", "What is your employee ID for ticket creation purposes?"),
    ("user",  "It's 10681."),
]

turns_right = [
    ("agent", "What is your employee email address associated with this issue?"),
    ("user",  "It's rudhik@gmail.com."),
    ("agent", "Could you please provide the location where the issue occurred?"),
    ("user",  "Hyderabad."),
    ("agent", "Here's the summary of your ticket:\nEmployee ID: 10681\nEmail: rudhik@gmail.com\nLocation: Hyderabad\nShort Description: Unable to log in to VDI\nDetailed Description: Credentials timeout, cannot see anything\nCategory: VDI  ·  Priority: Critical\n\nIs everything correct or would you like to change anything?"),
    ("user",  "Everything is correct."),
    ("agent", "Creating ticket... Ticket created successfully.\n Ticket ID: INC0010004"),
]

draw_bubbles(s_c1, turns_left,  START_Y, COL_L, COL_L, BW)
draw_bubbles(s_c1, turns_right, START_Y, COL_R, COL_R, BW)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE — CASE 2: USER REQUESTS A FIELD CHANGE
# ══════════════════════════════════════════════════════════════════════════════
s_c2 = prs.slides.add_slide(blank_layout)
add_bg(s_c2)
section_header(s_c2,
               "Case 2 — Field Correction  ·  User Requests to Change Employee ID",
               "Agent re-collects the corrected field and re-shows the updated summary before creating the ticket",
               "CASE 2", ACCENT_ORG)

turns2_left = [
    ("user",  "Hello. How can you help me today?"),
    ("agent", "Hi! I can take care of creating your ServiceNow ticket. Could you describe the issue you're facing?"),
    ("user",  "I'm not able to log in to my VDI."),
    ("agent", "Can you provide more details about the issue you're experiencing?"),
    ("user",  "After providing the credentials, it is taking too much time to log in."),
    ("agent", "What is your employee ID for ticket creation purposes?"),
    ("user",  "It's 1081."),
    ("agent", "What is your employee email address associated with this issue?"),
    ("user",  "It's rutik@gmail.com."),
    ("agent", "Could you please provide the location where the issue occurred?"),
    ("user",  "It's in Hyderabad."),
]

turns2_right = [
    ("agent", "Here's the summary of your ticket:\nEmployee ID: 1081\nEmail: rutik@gmail.com\nLocation: Hyderabad\nShort Description: Unable to log in to VDI\nDetailed Description: Credentials timeout on login\nCategory: VDI  ·  Priority: Critical\n\nIs everything correct or would you like to change anything?"),
    ("user",  "I want to change my employee ID to 10681."),
    ("agent", "Please provide new employee ID."),
    ("user",  "It's 10681."),
    ("agent", "Here's the updated summary:\nEmployee ID: 10681\nEmail: rutik@gmail.com\nLocation: Hyderabad\nShort Description: Unable to log in to VDI\nDetailed Description: Credentials timeout on login\nCategory: VDI  ·  Priority: Critical\n\nIs everything correct or would you like to change anything?"),
    ("user",  "Everything is correct."),
    ("agent", "Creating ticket... Ticket created successfully.\n Ticket ID: INC0010004"),
]

draw_bubbles(s_c2, turns2_left,  START_Y, COL_L, COL_L, BW)
draw_bubbles(s_c2, turns2_right, START_Y, COL_R, COL_R, BW)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE — CASE 3: UNRELATED / OUT-OF-SCOPE QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════
s_c3 = prs.slides.add_slide(blank_layout)
add_bg(s_c3)
section_header(s_c3,
               "Case 3 — Out-of-Scope Questions  ·  Graceful Redirection",
               "Agent classifies unrelated input via LangGraph 'unrelated' node and steers user back to ticket creation",
               "CASE 3", ACCENT_RED)

# Single centered column for shorter conversation
SINGLE_L = Inches(2.0)
SINGLE_W = Inches(9.33)

turns3 = [
    ("user",  "Hello, do you know about Java?"),
    ("agent", "Hi! I can take care of creating your ServiceNow ticket. Could you describe the issue you're facing?"),
    ("user",  "Go and drink some water for yourself."),
    ("agent", "Hi! I can take care of creating your ServiceNow ticket. Could you describe the issue you're facing?"),
]

draw_bubbles(s_c3, turns3, START_Y, SINGLE_L, SINGLE_L, SINGLE_W)

# LangGraph node annotation
add_rect(s_c3, Inches(2.0), Inches(5.5), Inches(9.33), Inches(1.75), CARD_BG, ACCENT_RED, Pt(1.2))
add_text_box(s_c3, "LangGraph Routing — How It Works", Inches(2.15), Inches(5.6), Inches(9.0), Inches(0.35),
             font_size=Pt(13), bold=True, color=ACCENT_RED)

node_flow = [
    ("classify()", ACCENT_PURP,   "Detects: greeting / ticket / unrelated"),
    ("greeting()", ACCENT_BLUE,   "Salutation → welcome message"),
    ("unrelated()", ACCENT_RED,   "Off-topic → redirect to ticket flow"),
    ("extract()",  ACCENT_GREEN,  "On-topic → begin collecting ticket fields"),
]
for i, (node, col, desc) in enumerate(node_flow):
    lx = Inches(2.15) + i * Inches(2.35)
    ty = Inches(6.02)
    add_rect(s_c3, lx, ty, Inches(1.9), Inches(0.35), col)
    add_text_box(s_c3, node, lx, ty + Inches(0.04), Inches(1.9), Inches(0.3),
                 font_size=Pt(10), bold=True, align=PP_ALIGN.CENTER)
    add_text_box(s_c3, desc, lx, ty + Inches(0.42), Inches(1.9), Inches(0.5),
                 font_size=Pt(8.5), color=LIGHT_GREY, align=PP_ALIGN.CENTER)

# ── Save ───────────────────────────────────────────────────────────────────────
prs.save(FILE)
print(f"Saved {FILE} with 3 conversation slides appended.")
