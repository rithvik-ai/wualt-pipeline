const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat, TabStopType, TabStopPosition,
} = require("docx");

// ── Colours ──────────────────────────────────────────────────
const C = {
  primary:   "1B2A4A",
  accent:    "2E75B6",
  accent2:   "4A90D9",
  green:     "2D8A4E",
  amber:     "C07D10",
  red:       "C0392B",
  headerBg:  "1B2A4A",
  headerTxt: "FFFFFF",
  rowAlt:    "F0F4FA",
  rowWhite:  "FFFFFF",
  lightBlue: "E8F0FE",
  lightGrey: "F5F6F8",
  border:    "B0B8C8",
  divider:   "2E75B6",
  textDark:  "1A1A2E",
  textMid:   "3A3A5C",
  textLight: "5A5A7A",
};

// ── Page dims ────────────────────────────────────────────────
const PAGE_W = 12240;
const PAGE_H = 15840;
const MARGIN = 1440;
const CONTENT_W = PAGE_W - 2 * MARGIN; // 9360

// ── Borders ──────────────────────────────────────────────────
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: C.border };
const borders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };
const noBorder = { style: BorderStyle.NONE, size: 0 };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const cellPad = { top: 80, bottom: 80, left: 120, right: 120 };

// ── Helpers ──────────────────────────────────────────────────
function hCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: C.headerBg, type: ShadingType.CLEAR },
    margins: cellPad,
    verticalAlign: "center",
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, font: "Arial", size: 20, color: C.headerTxt })] })],
  });
}

function dCell(text, width, alt = false, opts = {}) {
  const runs = [];
  if (opts.bold) {
    runs.push(new TextRun({ text, font: "Arial", size: 20, color: opts.color || C.textDark, bold: true }));
  } else {
    runs.push(new TextRun({ text, font: "Arial", size: 20, color: opts.color || C.textDark }));
  }
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: alt ? C.rowAlt : C.rowWhite, type: ShadingType.CLEAR },
    margins: cellPad,
    children: [new Paragraph({ children: runs })],
  });
}

function makeTable(headers, widths, rows) {
  const hdrRow = new TableRow({ children: headers.map((h, i) => hCell(h, widths[i])) });
  const dataRows = rows.map((row, ri) =>
    new TableRow({ children: row.map((cell, ci) => dCell(cell, widths[ci], ri % 2 === 1)) })
  );
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [hdrRow, ...dataRows],
  });
}

function spacer(pts = 120) {
  return new Paragraph({ spacing: { after: pts }, children: [] });
}

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, font: "Arial", size: 32, bold: true, color: C.primary })],
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: [new TextRun({ text, font: "Arial", size: 26, bold: true, color: C.accent })],
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, font: "Arial", size: 22, bold: true, color: C.textDark })],
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after || 120 },
    alignment: opts.align || AlignmentType.LEFT,
    children: [new TextRun({ text, font: "Arial", size: 22, color: opts.color || C.textDark, ...(opts.bold ? { bold: true } : {}), ...(opts.italic ? { italics: true } : {}) })],
  });
}

function multiPara(segments) {
  return new Paragraph({
    spacing: { after: 120 },
    children: segments.map(s => new TextRun({ text: s.text, font: "Arial", size: 22, color: s.color || C.textDark, bold: !!s.bold, italics: !!s.italic })),
  });
}

function divider() {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.divider, space: 1 } },
    children: [],
  });
}

function calloutBox(title, text) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({
      children: [new TableCell({
        borders: { top: noBorder, bottom: noBorder, right: noBorder, left: { style: BorderStyle.SINGLE, size: 12, color: C.accent } },
        width: { size: CONTENT_W, type: WidthType.DXA },
        shading: { fill: C.lightBlue, type: ShadingType.CLEAR },
        margins: { top: 120, bottom: 120, left: 200, right: 160 },
        children: [
          new Paragraph({ children: [new TextRun({ text: title, font: "Arial", size: 22, bold: true, color: C.accent })] }),
          new Paragraph({ spacing: { before: 60 }, children: [new TextRun({ text, font: "Arial", size: 20, color: C.textMid })] }),
        ],
      })],
    })],
  });
}

function codeBlock(text) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({
      children: [new TableCell({
        borders,
        width: { size: CONTENT_W, type: WidthType.DXA },
        shading: { fill: C.lightGrey, type: ShadingType.CLEAR },
        margins: { top: 100, bottom: 100, left: 200, right: 160 },
        children: [new Paragraph({ children: [new TextRun({ text, font: "Consolas", size: 18, color: C.textMid })] })],
      })],
    })],
  });
}

// ══════════════════════════════════════════════════════════════
// DOCUMENT
// ══════════════════════════════════════════════════════════════

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: C.primary },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: C.accent },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: C.textDark },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "bullets2", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers3", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "bullets3", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "bullets4", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "bullets5", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers4", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers5", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [
    // ══════════════════════════════════════════════════════════
    // TITLE PAGE
    // ══════════════════════════════════════════════════════════
    {
      properties: {
        page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } },
      },
      children: [
        spacer(2400),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "WUALT", font: "Arial", size: 72, bold: true, color: C.accent })] }),
        spacer(200),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Safety Intelligence Engine", font: "Arial", size: 36, color: C.primary })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Three-Layer Detection System", font: "Arial", size: 36, color: C.primary })] }),
        spacer(300),
        divider(),
        spacer(200),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Scientific Process, Rationale & Architecture", font: "Arial", size: 24, color: C.accent })] }),
        spacer(600),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Confidential \u2014 WUALT Team Internal", font: "Arial", size: 20, color: C.textLight })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "April 2026 \u00B7 Version 1.0", font: "Arial", size: 20, color: C.textLight })] }),
        new Paragraph({ children: [new PageBreak()] }),
      ],
    },

    // ══════════════════════════════════════════════════════════
    // MAIN CONTENT
    // ══════════════════════════════════════════════════════════
    {
      properties: {
        page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: C.accent, space: 4 } },
            tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
            children: [
              new TextRun({ text: "WUALT Safety Intelligence Engine", font: "Arial", size: 16, color: C.textLight }),
              new TextRun({ text: "\tScientific Process Document", font: "Arial", size: 16, color: C.textLight }),
            ],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            border: { top: { style: BorderStyle.SINGLE, size: 2, color: C.border, space: 4 } },
            tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
            children: [
              new TextRun({ text: "Confidential \u2014 WUALT", font: "Arial", size: 16, color: C.textLight }),
              new TextRun({ text: "\tPage ", font: "Arial", size: 16, color: C.textLight }),
              new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: C.textLight }),
            ],
          })],
        }),
      },
      children: [

        // ═══════════════════════════════════════════════════════
        // 1. INTRODUCTION & PURPOSE
        // ═══════════════════════════════════════════════════════
        heading1("1. Introduction & Purpose"),

        para("This document describes the scientific reasoning, physiological foundations, and engineering workflow behind the WUALT Safety Intelligence Engine \u2014 a three-layer real-time detection system for a wearable smart ring. Layer 1 (Physiological Distress) monitors four vital signals to detect stress and distress. Layer 2 (Fall Detection) uses a 3-stage accelerometer/gyroscope model to detect falls. Layer 3 (Geospatial Safety) combines location, movement, time, and connectivity context with physiological signals for comprehensive safety risk scoring."),

        heading2("1.1 The Problem"),
        para("Wearable devices generate continuous physiological data, but raw sensor values alone cannot determine whether a user is in danger. A heart rate of 140 bpm could mean the user is jogging (safe) or experiencing a panic attack (needs intervention). The challenge is to distinguish genuine physiological distress from normal physiological variation \u2014 in real time, with noisy consumer-grade sensors, across diverse user demographics."),

        heading2("1.2 Design Philosophy"),
        para("We chose a deterministic rule-based architecture over machine learning for this stage of the product. The reasons are scientific and regulatory:"),

        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "Interpretability: ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "Every detection decision traces to a specific threshold grounded in published clinical literature. A clinician can audit and verify any alert.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "Safety-first: ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "Rule-based systems have predictable failure modes. If a threshold is wrong, you change one number. ML models can fail unpredictably on unseen distributions.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "Regulatory simplicity: ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "Thresholds map directly to BTS, WHO, and peer-reviewed guidelines. No black-box model validation required.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 120 }, children: [
          new TextRun({ text: "Zero data dependency: ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "Works immediately without labelled training data. Thresholds are calibrated against published physiology and validated on synthetic datasets.", font: "Arial", size: 22, color: C.textDark }),
        ]}),

        spacer(),

        // ═══════════════════════════════════════════════════════
        // 2. PHYSIOLOGICAL FOUNDATIONS
        // ═══════════════════════════════════════════════════════
        heading1("2. Physiological Foundations"),
        para("The engine monitors four signals. Each was selected because it is measurable from a finger-worn PPG ring and has well-established relationships to the autonomic stress response."),

        heading2("2.1 Heart Rate (HR) \u2014 Weight: 0.35"),
        heading3("The Science"),
        para("When the body perceives threat (physical or psychological), the sympathetic nervous system activates the fight-or-flight response. The adrenal medulla releases adrenaline and noradrenaline, which act on beta-1 adrenergic receptors in the sinoatrial node to increase heart rate within seconds. This makes HR the fastest and most reliable indicator of acute stress."),
        para("Resting heart rate varies by individual (40\u201390 bpm depending on age, fitness, and genetics). Therefore, both personalized z-score thresholds and universal absolute thresholds are needed."),

        heading3("Threshold Justification"),
        makeTable(
          ["Threshold", "Value", "Source", "Rationale"],
          [2200, 1400, 2800, 2960],
          [
            ["Z-score", "> +2.0", "Statistical convention", "2 standard deviations above personal mean captures ~97.7th percentile \u2014 values this far above baseline are statistically unlikely under normal conditions"],
            ["Absolute (High)", "\u2265 120 bpm", "Clinical tachycardia criteria", "Sinus tachycardia threshold at rest; universally considered elevated regardless of individual baseline"],
            ["Absolute (Emergency)", "\u2265 150 bpm", "Tanaka et al. 2001 (MPHR formula)", "At rest, this approaches maximum predicted HR for many adults (208 \u2212 0.7 \u00D7 age). Bypasses persistence for immediate escalation"],
          ]
        ),

        calloutBox("Why 0.35 Weight (Highest)?", "HR is the most responsive signal to acute stress activation, changes within seconds, and is the most reliably measured by PPG sensors. It earns the highest weight because of its combination of physiological primacy and measurement reliability."),
        spacer(),

        heading2("2.2 Blood Oxygen Saturation (SpO2) \u2014 Weight: 0.30"),
        heading3("The Science"),
        para("SpO2 measures the percentage of haemoglobin binding sites occupied by oxygen. Normal range is 95\u2013100%. Desaturation can indicate respiratory compromise, airway obstruction, perfusion failure, or altitude-related hypoxia. Unlike HR (which responds to both psychological and physical stress), SpO2 drops are almost always clinically significant."),
        para("The ring measures SpO2 via dual-wavelength PPG (red and infrared). Accuracy degrades during motion, which is why the preprocessing pipeline computes an SpO2-specific SQI sub-score."),

        heading3("Threshold Justification"),
        makeTable(
          ["Threshold", "Value", "Source", "Rationale"],
          [2200, 1400, 2800, 2960],
          [
            ["Z-score", "< \u22122.0", "Statistical convention", "2 standard deviations below personal mean; catches personal desaturation that may be within \u201cnormal\u201d population range"],
            ["Absolute (Concern)", "\u2264 95%", "British Thoracic Society (BTS) 2017", "Normal SpO2 is 95\u2013100%. Values at or below 95% indicate clinical concern requiring monitoring"],
            ["Absolute (Emergency)", "\u2264 90%", "WHO definition of hypoxemia", "Below 90% is medical emergency. Bypasses all persistence requirements for immediate distress escalation"],
          ]
        ),

        calloutBox("Why Never Suppress SpO2 During Exercise?", "Unlike HR (which is expected to rise during exercise), SpO2 should remain stable or decrease only minimally during physical activity. A significant SpO2 drop during exercise could indicate exercise-induced bronchoconstriction, cardiac shunting, or altitude sickness. The engine NEVER suppresses SpO2 flags regardless of motion state."),
        spacer(),

        calloutBox("Clinical Floor Guard for Z-Score SpO2", "The z-score SpO2 flag includes a clinical floor check: z-score deviation only triggers when the actual SpO2 value is below 96%. Values 96\u2013100% are clinically normal and should not alarm the user even if they deviate significantly from a high personal baseline (e.g., a user with baseline 98.5 \u00B1 0.5 showing SpO2 of 97%). This prevents false \u201Coxygen is low\u201D alerts for healthy individuals with naturally high baselines."),
        spacer(),

        heading2("2.3 Heart Rate Variability (HRV RMSSD) \u2014 Weight: 0.25"),
        heading3("The Science"),
        para("HRV (specifically RMSSD \u2014 Root Mean Square of Successive Differences in R-R intervals) reflects the balance between sympathetic and parasympathetic branches of the autonomic nervous system. High RMSSD indicates strong parasympathetic (vagal) tone \u2014 the body is relaxed and adaptive. Suppressed RMSSD indicates sympathetic dominance \u2014 the classic stress response."),
        para("HRV is a well-established biomarker in clinical literature (Shaffer & Ginsberg 2017, Thayer et al. 2012) for chronic stress, anxiety, and autonomic dysfunction. Unlike HR (which captures acute stress), HRV captures sustained autonomic imbalance."),

        heading3("Threshold Justification"),
        makeTable(
          ["Threshold", "Value", "Source", "Rationale"],
          [2200, 1400, 2800, 2960],
          [
            ["Z-score", "< \u22121.5", "More sensitive than \u22122.0", "HRV responds to chronic/sustained stress, not just acute. A lower threshold (\u22121.5) catches subtle autonomic dysregulation that \u22122.0 would miss"],
            ["Absolute", "None", "N/A \u2014 too individual", "HRV varies enormously: athletic adults may have RMSSD of 65 ms, elderly sedentary adults 18 ms. No universal \u201cbad HRV\u201d number exists"],
          ]
        ),

        calloutBox("Why Lower Weight (0.25) Than HR?", "Ring-based PPG HRV is inherently less accurate than chest-strap ECG (Esco & Flatt 2014). The preprocessing pipeline caps HRV SQI at 0.7 to reflect this limitation. HRV earns a lower weight because measurement noise is higher, but it remains valuable as a complementary autonomic indicator."),
        spacer(),

        heading2("2.4 Skin Temperature \u2014 Weight: 0.10"),
        heading3("The Science"),
        para("The ring measures skin temperature at the finger, which is 4\u20137\u00B0C below core temperature. Skin temperature is influenced by peripheral vasodilation/vasoconstriction (autonomic response), ambient environment, clothing, post-meal thermogenesis, circadian rhythm, and physical activity."),
        para("This makes skin temperature a poor standalone indicator of distress. However, when combined with other signals, elevated skin temperature supports detection. For example: elevated HR + elevated skin temp + suppressed HRV is more likely genuine fever/infection than HR elevation alone."),

        heading3("Threshold Justification"),
        makeTable(
          ["Threshold", "Value", "Source", "Rationale"],
          [2200, 1400, 2800, 2960],
          [
            ["Z-score", "> +2.5", "High threshold (deliberate)", "Because skin temp fluctuates with environment, we require a very large personal deviation (2.5 std devs) before considering it meaningful"],
            ["Absolute", "\u2265 37.8\u00B0C", "Fever screening literature", "Skin temp of 37.8\u00B0C corresponds roughly to core temp of 38.5\u201339.5\u00B0C (accounting for 4\u20137\u00B0C offset), indicating possible fever"],
          ]
        ),

        calloutBox("Critical Design Decision: Temp Alone Never Triggers Stress", "Temperature is classified as a \u201Cweak/supporting\u201D signal. Even if skin temp exceeds threshold, if it is the ONLY flagged signal, the engine state remains \u201Cnormal\u201D with an advisory message. This prevents false alarms from ambient heat, hot beverages, or warm rooms."),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 3. DUAL-THRESHOLD DETECTION SYSTEM
        // ═══════════════════════════════════════════════════════
        heading1("3. The Dual-Threshold Detection System"),
        para("A single threshold system cannot handle both personalisation and safety. The engine uses two complementary systems operating in parallel:"),

        heading2("3.1 Z-Score Thresholds (Personalised)"),
        para("The preprocessing pipeline maintains a rolling baseline of each signal over approximately 300 frames (~5 minutes). Z-scores are computed as:"),
        codeBlock("z = (current_value \u2212 baseline_mean) / baseline_std"),
        spacer(60),
        para("This personalises detection to each individual. An athlete with resting HR of 48 bpm who suddenly reaches 80 bpm has z \u2248 +8.0 and would be flagged, even though 80 bpm is \u201Cnormal\u201D for most adults. Conversely, a sedentary adult with resting HR of 88 bpm at 95 bpm has z \u2248 +1.2 and would NOT be flagged."),

        heading2("3.2 Absolute Thresholds (Universal Safety Net)"),
        para("Always active, including during cold-start before any baseline is established. These are hard physiological limits derived from clinical guidelines that apply regardless of individual variation:"),

        makeTable(
          ["Signal", "Threshold", "Guideline Source"],
          [2400, 3400, 3560],
          [
            ["HR", "\u2265 120 bpm", "Clinical sinus tachycardia definition"],
            ["HR", "\u2265 150 bpm (emergency)", "Tanaka 2001 MPHR margins"],
            ["SpO2", "\u2264 95%", "British Thoracic Society 2017"],
            ["SpO2", "\u2264 90% (emergency)", "WHO hypoxemia definition"],
            ["Skin Temp", "\u2265 37.8\u00B0C", "Fever screening literature"],
          ]
        ),

        calloutBox("Why Both Systems?", "Z-scores catch subtle personal deviations. Absolute thresholds catch emergencies that z-scores might miss (e.g., during cold-start, or if baseline drifted). Together they form a layered defence: personalised sensitivity backed by universal safety."),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 4. THREE-STATE MODEL
        // ═══════════════════════════════════════════════════════
        heading1("4. The Three-State Classification Model"),
        para("The engine classifies each evaluation into one of three states based on how many primary signal channels are simultaneously flagged:"),

        makeTable(
          ["State", "Flag Count", "Severity", "Scientific Rationale"],
          [1400, 1600, 1200, 5160],
          [
            ["Normal", "0 flags (or temp only)", "Low", "All signals within personal baseline. If only temperature is elevated, it is treated as advisory because skin temp is too unreliable as a standalone indicator."],
            ["Stress", "1 primary flag", "Low\u2013Medium", "A single physiological system is deviating. This could represent genuine stress, but also has many benign explanations (caffeine, postural change, anxiety). Advisory-level intervention."],
            ["Distress", "2+ primary flags", "High", "Multiple physiological systems are simultaneously disrupted. The probability of a benign explanation drops significantly when HR, SpO2, and/or HRV all deviate together. Urgent intervention suggested."],
          ]
        ),

        heading2("4.1 Why 2+ Signals for Distress?"),
        para("This follows the principle of converging evidence in clinical assessment. Any single signal can deviate for benign reasons:"),

        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: "HR elevated alone: could be caffeine, standing up, mild anxiety, dehydration", font: "Arial", size: 22, color: C.textDark })] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: "SpO2 dipped alone: could be transient sensor error, breath-holding, sleeping posture", font: "Arial", size: 22, color: C.textDark })] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: "HRV suppressed alone: could be time-of-day variation, digestion, mild fatigue", font: "Arial", size: 22, color: C.textDark })] }),

        para("When two or more systems show simultaneous deviation, the probability of a benign explanation decreases significantly. HR elevated AND SpO2 dropping suggests genuine cardiovascular or respiratory compromise, not just coffee."),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 5. EXERCISE SUPPRESSION
        // ═══════════════════════════════════════════════════════
        heading1("5. Motion-Aware Exercise Suppression"),
        para("Without exercise suppression, every jog would trigger continuous distress alerts. The engine solves this through accelerometer-based motion classification."),

        heading2("5.1 Motion Classification"),
        para("Dynamic acceleration magnitude (dyn_acc_mag) isolates the movement component by subtracting gravitational baseline from total acceleration:"),
        makeTable(
          ["Motion State", "Threshold (g)", "Interpretation"],
          [2400, 2400, 4560],
          [
            ["Still", "< 0.05", "Sitting, lying, standing \u2014 no movement"],
            ["Walking", "0.05 \u2013 0.25", "Casual walking, light movement"],
            ["Active", "0.25 \u2013 0.50", "Brisk walking, light sport"],
            ["Exercise", "> 0.50", "Running, HIIT, vigorous physical activity"],
          ]
        ),

        heading2("5.2 The Suppression Rule"),
        para("When motion is \u201Cactive\u201D or \u201Cexercise\u201D:"),

        new Paragraph({ numbering: { reference: "bullets2", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "HR flag is CLEARED ", font: "Arial", size: 22, bold: true, color: C.green }),
          new TextRun({ text: "if HR is the only flagged signal and HR < 150 bpm. Elevated heart rate during exercise is expected physiology.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "bullets2", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "HR flag is KEPT ", font: "Arial", size: 22, bold: true, color: C.red }),
          new TextRun({ text: "if SpO2 is also flagged. A runner with HR=145 and SpO2=91% may be in real danger \u2014 exercise cannot explain the oxygen drop.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "bullets2", level: 0 }, spacing: { after: 120 }, children: [
          new TextRun({ text: "HR flag is KEPT ", font: "Arial", size: 22, bold: true, color: C.red }),
          new TextRun({ text: "if HR \u2265 150 bpm regardless of exercise. Even during vigorous activity, HR above 150 bpm approaches maximum predicted HR and warrants monitoring.", font: "Arial", size: 22, color: C.textDark }),
        ]}),

        heading2("5.3 Why NOT Suppress HRV During Exercise?"),
        para("HRV naturally drops during exercise, but extreme HRV suppression during physical activity can indicate overtraining, cardiac dysfunction, or autonomic exhaustion. Suppressing HRV flags during exercise would remove a clinically meaningful signal. HRV flags stand during exercise as a secondary indicator."),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 6. PERSISTENCE MECHANISM
        // ═══════════════════════════════════════════════════════
        heading1("6. Persistence Mechanism"),
        para("A single-frame anomaly must NOT trigger an alert. Coughing, sneezing, a sudden arm movement, or sensor contact loss can all produce momentary signal spikes. The persistence requirement ensures that conditions are sustained before the engine promotes its state."),

        heading2("6.1 How It Works"),
        makeTable(
          ["Parameter", "Value", "Purpose"],
          [2800, 1800, 4760],
          [
            ["PERSISTENCE_STRESS_S", "60 seconds", "Stress condition must hold continuously for 60 seconds before the user sees a stress alert"],
            ["PERSISTENCE_DISTRESS_S", "60 seconds", "Distress condition must hold 60 seconds before escalation"],
            ["Grace period", "2 frames", "If condition drops for 1\u20132 frames (sensor noise), the timer does NOT reset. Only resets after 3+ clear frames"],
          ]
        ),

        heading2("6.2 Scientific Rationale"),
        para("The 60-second window is informed by the temporal dynamics of genuine physiological stress responses:"),

        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: "Startle reflex: HR spikes for 5\u201315 seconds, then recovers (Vila et al. 2007)", font: "Arial", size: 22, color: C.textDark })] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: "Cough/sneeze: Transient physiological disruption lasting < 10 seconds", font: "Arial", size: 22, color: C.textDark })] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: "Genuine stress response: Sympathetic activation sustains for minutes to hours", font: "Arial", size: 22, color: C.textDark })] }),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 120 }, children: [new TextRun({ text: "Motion artifact: Accelerometer spike resolves within 1\u20133 frames", font: "Arial", size: 22, color: C.textDark })] }),

        para("The 60-second threshold filters out all transient events while remaining responsive enough to detect genuine episodes within a clinically appropriate timeframe."),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 7. EMERGENCY BYPASS
        // ═══════════════════════════════════════════════════════
        heading1("7. Emergency Bypass Logic"),
        para("Certain conditions are so dangerous that waiting 60 seconds is clinically unacceptable. Two emergency rules bypass all persistence requirements:"),

        makeTable(
          ["Emergency Condition", "Action", "Clinical Justification"],
          [2800, 2400, 4160],
          [
            ["SpO2 \u2264 90%", "Immediate DISTRESS", "WHO defines SpO2 < 90% as hypoxemia. Brain damage can begin within minutes of sustained hypoxia. Delay is not acceptable."],
            ["HR \u2265 150 at rest", "Immediate STRESS (minimum)", "Resting HR at 150+ bpm approaches maximum predicted heart rate for many adults (Tanaka 2001). At rest, this suggests SVT, panic disorder, or cardiac emergency."],
          ]
        ),

        calloutBox("Exercise Exception for HR Emergency", "The HR \u2265 150 bypass does NOT fire during exercise. HR of 160 bpm while sprinting is physiologically normal; HR of 160 bpm while sitting still is a medical emergency. Motion context distinguishes these."),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 8. CONFIDENCE SCORING
        // ═══════════════════════════════════════════════════════
        heading1("8. Confidence Scoring Model"),
        para("Every detection output includes a confidence score (0.0\u20131.0) that quantifies the engine\u2019s certainty in its current assessment."),

        heading2("8.1 Normal State Confidence"),
        codeBlock("confidence = SQI_overall \u00D7 0.6  +  window_factor \u00D7 0.4"),
        spacer(60),
        para("Higher signal quality and more frames in the rolling window increase confidence in normality. After ~20 frames of stable data with good SQI, normal confidence reaches ~0.93."),

        heading2("8.2 Stress/Distress State Confidence"),
        codeBlock("confidence = signal_score \u00D7 0.5  +  persist_factor \u00D7 0.3  +  SQI_factor \u00D7 0.2"),
        spacer(60),

        makeTable(
          ["Component", "Weight", "Formula", "Rationale"],
          [1800, 1200, 3400, 2960],
          [
            ["Signal score", "50%", "Weighted sum of flagged signals", "Stronger signal deviations produce higher confidence"],
            ["Persistence", "30%", "Ramps 0.3 \u2192 1.0 over 120s", "A condition sustained for 2 minutes is far more reliable than one that just appeared"],
            ["SQI factor", "20%", "min(1.0, SQI / 0.8)", "Poor data quality reduces confidence in ANY detection"],
          ]
        ),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 9. SIGNAL WEIGHT RATIONALE
        // ═══════════════════════════════════════════════════════
        heading1("9. Signal Weight Rationale"),
        para("The weights (HR=0.35, SpO2=0.30, HRV=0.25, Temp=0.10) are not arbitrary. They reflect three factors:"),

        makeTable(
          ["Factor", "HR (0.35)", "SpO2 (0.30)", "HRV (0.25)", "Temp (0.10)"],
          [2000, 1840, 1840, 1840, 1840],
          [
            ["Physiological relevance", "Highest \u2014 primary acute stress marker", "Critical safety signal", "Chronic stress marker", "Weak \u2014 affected by environment"],
            ["Measurement reliability (ring)", "High \u2014 PPG HR is accurate", "Moderate \u2014 degrades with motion", "Low \u2014 ring PPG less accurate than ECG", "Moderate \u2014 ambient interference"],
            ["Responsiveness", "Seconds", "Seconds to minutes", "Minutes to hours", "Minutes to hours"],
            ["Standalone diagnostic value", "High", "Very high (always clinical)", "Moderate", "Low"],
          ]
        ),

        para("The sum of weights equals 1.0, forming a normalised confidence scoring basis. Temperature\u2019s low weight (0.10) ensures it contributes minimally to confidence unless it accompanies primary signals."),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 10. CLINICAL FLAG ARCHITECTURE
        // ═══════════════════════════════════════════════════════
        heading1("10. Clinical vs. Quality Flag Architecture"),
        para("A fundamental design decision separates two types of flags:"),

        makeTable(
          ["Property", "Reject Reasons (Quality)", "Clinical Flags (Advisory)"],
          [2200, 3580, 3580],
          [
            ["Purpose", "Sensor data is unreliable", "User health concern detected"],
            ["Effect on frame", "accepted = False \u2014 frame discarded", "accepted unchanged \u2014 frame still analysed"],
            ["Effect on detection", "Engine skips evaluation entirely", "Flags integrated into signal detection"],
            ["Examples", "finger_off, low_ppg_variance, hr_out_of_range", "spo2_clinical_concern (93%), elevated_skin_temp (37.9\u00B0C)"],
            ["Rationale", "Bad data leads to bad decisions", "A sick user with clean data needs monitoring more, not less"],
          ]
        ),

        calloutBox("Why This Matters", "If clinical concerns (e.g., fever) were mixed with quality rejections, a sick user\u2019s frames would be discarded precisely when monitoring is most important. The two-list architecture ensures that data quality and health status are evaluated independently."),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 11. COLD-START & BASELINE WARMUP
        // ═══════════════════════════════════════════════════════
        heading1("11. Cold-Start & Baseline Warmup"),
        para("When the ring is first put on (or after a long gap), the preprocessing pipeline needs ~30\u201360 seconds to establish a stable statistical baseline."),

        heading2("11.1 Behaviour During Cold-Start"),
        new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: "baseline_ready = False \u2014 z-score thresholds are disabled (all z-scores = 0.0)", font: "Arial", size: 22, color: C.textDark })] }),
        new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: "Only absolute thresholds are active \u2014 the universal safety net", font: "Arial", size: 22, color: C.textDark })] }),
        new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: "User alert: \u201CWe\u2019re still learning your baseline. Alerts will become more personalised shortly.\u201D", font: "Arial", size: 22, color: C.textDark })] }),
        new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { after: 120 }, children: [new TextRun({ text: "Confidence is lower because the window factor is small (few frames accumulated)", font: "Arial", size: 22, color: C.textDark })] }),

        heading2("11.2 Safety During Cold-Start"),
        para("If someone puts on the ring and immediately has SpO2 at 88%, the absolute threshold catches it and fires immediate distress \u2014 no baseline needed. The safety net is always active from frame one."),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 12. ALERT SYSTEM
        // ═══════════════════════════════════════════════════════
        heading1("12. STAR-Principle Alert System"),
        para("All user-facing messages follow the STAR principle (Situation, Task, Action, Result) internally, but the output is natural, conversational language with no medical jargon."),

        heading2("12.1 Alert Design Principles"),
        new Paragraph({ numbering: { reference: "numbers3", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "No medical jargon: ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "\u201Coxygen level\u201D not \u201CSpO2 saturation\u201D; \u201Cheart rate\u201D not \u201Ctachycardia\u201D", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers3", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "No alarming language: ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "\u201Cseems a bit stressed\u201D not \u201CWARNING: abnormal vitals detected\u201D", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers3", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "Actionable suggestions: ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "\u201CTry some slow, deep breaths\u201D; \u201CReach out to someone you trust\u201D", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers3", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "No diagnosis: ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "\u201Csigns of distress\u201D not \u201Cyou may be having a panic attack\u201D", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers3", level: 0 }, spacing: { after: 120 }, children: [
          new TextRun({ text: "Message rotation: ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "Multiple variants per category to prevent repetition during sustained conditions", font: "Arial", size: 22, color: C.textDark }),
        ]}),

        heading2("12.2 Alert Categories"),
        makeTable(
          ["Category", "Severity", "Example Message"],
          [2200, 1400, 5760],
          [
            ["Normal", "Low", "Everything looks steady. You\u2019re doing great."],
            ["Warming up", "Low", "We\u2019re still learning your baseline."],
            ["Temp advisory", "Low", "Your skin temperature is a bit warmer than usual, but there\u2019s no need to worry."],
            ["Stress (HR)", "Low", "Your heart rate is a bit higher than usual. If you\u2019re not exercising, try to relax."],
            ["Stress (SpO2)", "Medium", "Your oxygen level dipped a little. Try some slow, deep breaths."],
            ["Exercise elevated", "Low", "Your readings are elevated, but you seem to be moving \u2014 likely physical activity."],
            ["Distress", "High", "We\u2019re noticing signs of distress. If you\u2019re not feeling safe, please reach out."],
            ["Distress (SpO2)", "High", "Your oxygen level is lower than expected. If feeling lightheaded, please seek help."],
          ]
        ),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 13. END-TO-END WORKFLOW
        // ═══════════════════════════════════════════════════════
        heading1("13. End-to-End Workflow"),
        para("This section describes the complete data flow from raw ring sensor data to user-facing alert, and the development/calibration workflow."),

        heading2("13.1 Real-Time Data Flow (Per Frame)"),

        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Ring Hardware ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Samples PPG, accelerometer, thermometer at 1 Hz. Transmits raw frame via BLE.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Preprocessing Pipeline ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Receives raw frame. Applies EMA smoothing (\u03B1=0.5), thermal bias correction, SQI computation (7 sub-scores), dynamic acceleration extraction, z-score computation against rolling baseline, clinical flag generation. Outputs pipeline_output dict.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Quality Gate ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Checks frame acceptance and SQI \u2265 0.5. Rejected or low-quality frames are skipped; persistence resets.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Motion Classification ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Classifies dynamic acceleration into still/walking/active/exercise.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Signal Flagging ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Evaluates each of 4 signals against z-score thresholds (personalized), absolute thresholds (universal), and clinical flags (advisory). Produces per-signal boolean flags and severity scores.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Exercise Suppression ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 If exercising and only HR is flagged (no SpO2 concern, HR < 150), clears the HR flag.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "State Determination ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Maps flag count to raw state: 0 flags = normal, 1 primary flag = stress, 2+ flags = distress. Temp-only stays normal.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Persistence Check ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Condition must persist 60s before promotion. Emergency signals (SpO2 \u2264 90%, HR \u2265 150 at rest) bypass this requirement.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Confidence Scoring ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Computes 0.0\u20131.0 confidence from weighted signal scores, persistence duration, and data quality.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 120 }, children: [
          new TextRun({ text: "Alert Selection ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Picks the most appropriate message based on state, dominant signal, motion context, and baseline readiness. Returns final result to the user interface.", font: "Arial", size: 22, color: C.textDark }),
        ]}),

        heading2("13.2 Development & Calibration Workflow"),
        new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Define user profiles ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "\u2014 15 demographic profiles with realistic resting baselines (athletes, elderly, COPD, pregnant, anxiety disorder)", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Define scenario templates ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "\u2014 37 physiological scenarios with ground-truth labels (7 normal, 4 exercise, 6 stress, 8 distress, 10 edge cases, 2 transitions)", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Generate synthetic dataset ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "\u2014 9,926 labelled frames across all profiles and scenarios with realistic noise and physiological dynamics", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Run validation ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "\u2014 Feed every frame through the engine, compare predicted vs expected state, compute precision/recall/F1 per class", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Identify calibration gaps ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "\u2014 Per-scenario accuracy breakdown reveals exactly which thresholds need tuning", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Adjust thresholds ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "\u2014 Modify z-score thresholds, absolute limits, motion boundaries, or signal weights in distress_engine.py", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { after: 100 }, children: [
          new TextRun({ text: "Re-validate ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "\u2014 Re-run generate_synthetic_dataset.py to see the impact. Iterate until desired accuracy.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { after: 120 }, children: [
          new TextRun({ text: "Deploy with real data ", font: "Arial", size: 22, bold: true, color: C.textDark }),
          new TextRun({ text: "\u2014 When real ring data becomes available, replace synthetic profiles with actual user baselines and repeat validation.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 14. VALIDATION RESULTS
        // ═══════════════════════════════════════════════════════
        heading1("14. Current Validation Results"),
        para("Running all 9,926 synthetic frames through the engine with persistence disabled (per-frame evaluation):"),

        makeTable(
          ["Metric", "Normal", "Stress", "Distress", "Rejected"],
          [1800, 1890, 1890, 1890, 1890],
          [
            ["Precision", "94.6%", "49.0%", "48.3%", "100.0%"],
            ["Recall", "60.9%", "45.9%", "93.9%", "100.0%"],
            ["F1 Score", "74.1%", "47.4%", "63.8%", "100.0%"],
          ]
        ),

        heading2("14.1 Key Findings"),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "Distress recall is 93.9% ", font: "Arial", size: 22, bold: true, color: C.green }),
          new TextRun({ text: "\u2014 the engine catches nearly all dangerous conditions. For a safety device, this is the most critical metric.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "Normal precision is 94.6% ", font: "Arial", size: 22, bold: true, color: C.green }),
          new TextRun({ text: "\u2014 when the engine says \u201Cnormal,\u201D it is almost always correct.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "Exercise false positives remain ", font: "Arial", size: 22, bold: true, color: C.amber }),
          new TextRun({ text: "\u2014 vigorous exercise scenarios sometimes have dyn_acc below 0.50g (noise), preventing exercise classification and HR suppression.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 120 }, children: [
          new TextRun({ text: "Stress-distress boundary blurs ", font: "Arial", size: 22, bold: true, color: C.amber }),
          new TextRun({ text: "\u2014 HR elevation naturally suppresses HRV, so some single-signal stress scenarios produce 2 flags (HR + HRV), overshooting to distress.", font: "Arial", size: 22, color: C.textDark }),
        ]}),

        para("These findings directly inform the next calibration cycle: lower the exercise motion threshold and consider HR-HRV correlation adjustment.", { italic: true }),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 15. FALL DETECTION (LAYER 2)
        // ═══════════════════════════════════════════════════════
        heading1("15. Fall Detection \u2014 Layer 2"),
        para("The second detection layer is a 3-stage biomechanical fall detection model using the ring\u2019s accelerometer and gyroscope data. Falls are one of the most dangerous events for wearable safety devices to detect, particularly for elderly users, those with mobility impairments, and women in distress situations."),

        heading2("15.1 3-Stage Detection Model"),
        para("Unlike naive threshold-based fall detectors (which suffer from high false-positive rates), the WUALT fall detector requires ALL three stages to occur in temporal sequence:"),

        makeTable(
          ["Stage", "Condition", "Threshold", "Rationale"],
          [1400, 2600, 1800, 3560],
          [
            ["1. Free-fall", "acc_mag < threshold", "< 0.5g", "During a fall, the body enters brief free-fall where effective acceleration drops below 1g. 0.5g threshold catches genuine falls while filtering stumbles"],
            ["2. Impact", "acc_mag > threshold AND orientation change > 60\u00B0", "> 3.0g, > 60\u00B0", "Impact with ground produces high-g spike. Orientation change confirms body position changed (not just a bump or clap)"],
            ["3. Post-fall inactivity", "dyn_acc < threshold for 3+ seconds", "< 0.05g for 3s", "Person remains motionless after impact \u2014 suggests injury, unconsciousness, or inability to move"],
          ]
        ),

        heading2("15.2 Temporal Constraints"),
        makeTable(
          ["Parameter", "Value", "Purpose"],
          [2800, 1800, 4760],
          [
            ["FALL_WINDOW_S", "5.0 seconds", "Maximum time between free-fall detection and impact. Falls happen fast \u2014 if impact doesn\u2019t occur within 5s, the free-fall was something else"],
            ["FALL_INACTIVITY_S", "3.0 seconds", "How long inactivity must persist to confirm. Brief stillness could be a pause; 3s sustained stillness after impact suggests incapacitation"],
            ["FALL_CONFIRM_TIMEOUT_S", "10.0 seconds", "If no inactivity confirmation within 10s of impact, the event is cancelled. Person got up \u2014 likely not a serious fall"],
          ]
        ),

        heading2("15.3 State Machine"),
        para("The detector operates as a finite state machine with four states:"),

        new Paragraph({ numbering: { reference: "numbers4", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "IDLE ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Monitoring for free-fall. Records pre-fall orientation for each frame.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers4", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "FREEFALL_DETECTED ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Free-fall confirmed. Waiting for high-g impact + orientation change within 5s window.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers4", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "IMPACT_DETECTED ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Impact confirmed. Monitoring for 3s of post-fall inactivity within 10s timeout.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "numbers4", level: 0 }, spacing: { after: 120 }, children: [
          new TextRun({ text: "CONFIRMED / CANCELLED ", font: "Arial", size: 22, bold: true, color: C.accent }),
          new TextRun({ text: "\u2192 Terminal states. CONFIRMED triggers emergency alerts; CANCELLED resets to IDLE.", font: "Arial", size: 22, color: C.textDark }),
        ]}),

        calloutBox("Why 3 Stages Instead of 1?", "Single-threshold fall detectors (e.g., \u201Cimpact > 3g = fall\u201D) produce excessive false positives from clapping hands, setting the ring on a table, or hitting a door. The 3-stage model requires a biomechanically plausible sequence: free-fall \u2192 impact + rotation \u2192 inactivity. This dramatically reduces false positives while maintaining sensitivity to real falls."),
        spacer(),

        calloutBox("Orientation Bug Fix", "An early implementation bug set the pre-fall orientation AFTER checking for free-fall, causing the orientation comparison to measure angle change within the fall itself (always small) rather than pre-fall vs. post-impact (large). The fix ensures pre-fall orientation is captured BEFORE the free-fall check, and only updated during non-freefall frames."),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 16. GEOSPATIAL SAFETY (LAYER 3)
        // ═══════════════════════════════════════════════════════
        heading1("16. Geospatial Safety Context \u2014 Layer 3"),
        para("The third detection layer integrates environmental context with physiological signals to produce a unified safety risk score. Context acts as a MODIFIER \u2014 it amplifies risk in dangerous situations and suppresses false alarms in safe environments."),

        heading2("16.1 Core Principle: Context as Modifier"),
        para("Context never triggers alerts independently. A user walking alone at 2 AM in an unfamiliar area with no phone is not inherently at risk \u2014 but if their physiological signals also show elevated distress, the context dramatically increases the urgency."),
        codeBlock("risk = physio_base \u00D7 max(0.3, confidence) \u00D7 context_amplifier"),
        spacer(60),
        para("The context amplifier ranges from 0.8\u00D7 (safe context suppresses borderline alerts) to 1.5\u00D7 (dangerous context escalates genuine concerns)."),

        heading2("16.2 Five Context Dimensions"),
        makeTable(
          ["Dimension", "Weight", "Low Risk Example", "High Risk Example"],
          [1800, 1200, 3180, 3180],
          [
            ["Time of Day", "25%", "Weekday afternoon (0.05)", "Late night 1\u20134 AM (0.85)"],
            ["Location", "30%", "Home zone (0.0)", "Unfamiliar area (0.7)"],
            ["Movement Pattern", "25%", "Normal walking (0.0)", "Erratic/running pattern (0.9)"],
            ["Connectivity", "10%", "Phone nearby + WiFi (0.0)", "No phone connection (0.6)"],
            ["Familiarity", "10%", "Visited 50+ times (0.0)", "Never visited (0.5)"],
          ]
        ),

        heading2("16.3 Context Score Computation"),
        para("Each dimension produces a score from 0.0 to 1.0. The weighted sum becomes the raw context score, which maps to the context amplifier:"),
        makeTable(
          ["Raw Score Range", "Amplifier", "Interpretation"],
          [2800, 2200, 4360],
          [
            ["0.00 \u2013 0.20", "0.80\u00D7 \u2013 0.90\u00D7", "Safe context \u2014 suppresses borderline alerts"],
            ["0.20 \u2013 0.40", "0.90\u00D7 \u2013 1.00\u00D7", "Neutral context \u2014 no modification"],
            ["0.40 \u2013 0.60", "1.00\u00D7 \u2013 1.15\u00D7", "Concerning context \u2014 mild amplification"],
            ["0.60 \u2013 0.80", "1.15\u00D7 \u2013 1.35\u00D7", "Dangerous context \u2014 significant amplification"],
            ["0.80 \u2013 1.00", "1.35\u00D7 \u2013 1.50\u00D7", "Critical context \u2014 maximum amplification"],
          ]
        ),

        heading2("16.4 5-Level Risk Classification"),
        makeTable(
          ["Level", "Score Range", "Response"],
          [2200, 2200, 4960],
          [
            ["Normal", "0.00 \u2013 0.30", "No action required"],
            ["Low Risk", "0.30 \u2013 0.50", "Ambient awareness \u2014 subtle notification"],
            ["Moderate Risk", "0.50 \u2013 0.70", "Active monitoring with contextual guidance"],
            ["High Risk", "0.70 \u2013 0.85", "Safety recommendations and emergency contact prep"],
            ["Critical", "0.85 \u2013 1.00", "Immediate intervention \u2014 auto-trigger emergency contacts after 30s persistence"],
          ]
        ),

        heading2("16.5 Suppression Rules (False Alarm Reduction)"),
        para("When the user is in a safe, familiar environment, borderline physiological signals should not cause unnecessary alerts:"),

        new Paragraph({ numbering: { reference: "bullets3", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "Home zone: ", font: "Arial", size: 22, bold: true, color: C.green }),
          new TextRun({ text: "Risk score \u00D7 0.5 \u2014 being at home dramatically reduces false alarm rates", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "bullets3", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "Work zone: ", font: "Arial", size: 22, bold: true, color: C.green }),
          new TextRun({ text: "Risk score \u00D7 0.6 \u2014 workplace is generally safe with people nearby", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "bullets3", level: 0 }, spacing: { after: 120 }, children: [
          new TextRun({ text: "Known area + walking: ", font: "Arial", size: 22, bold: true, color: C.green }),
          new TextRun({ text: "Risk score \u00D7 0.7 \u2014 familiar territory with normal movement", font: "Arial", size: 22, color: C.textDark }),
        ]}),

        heading2("16.6 Escalation Patterns (Women\u2019s Safety)"),
        para("Certain dangerous combinations force immediate escalation regardless of physiological baseline:"),

        new Paragraph({ numbering: { reference: "bullets4", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "Distress + night + unfamiliar + no phone: ", font: "Arial", size: 22, bold: true, color: C.red }),
          new TextRun({ text: "Forces risk score to 0.90 (critical). This combination strongly suggests a dangerous situation.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "bullets4", level: 0 }, spacing: { after: 80 }, children: [
          new TextRun({ text: "Route deviation + distress: ", font: "Arial", size: 22, bold: true, color: C.red }),
          new TextRun({ text: "If the user deviates significantly from their expected route while showing physiological distress, risk is escalated by +0.20.", font: "Arial", size: 22, color: C.textDark }),
        ]}),
        new Paragraph({ numbering: { reference: "bullets4", level: 0 }, spacing: { after: 120 }, children: [
          new TextRun({ text: "Phone lost + unfamiliar area: ", font: "Arial", size: 22, bold: true, color: C.red }),
          new TextRun({ text: "Losing phone connection in an unfamiliar area adds +0.15 to risk. The ring becomes the user\u2019s only safety device.", font: "Arial", size: 22, color: C.textDark }),
        ]}),

        heading2("16.7 Privacy-First Architecture"),
        para("The geospatial layer never stores raw GPS coordinates. All location processing converts to zone booleans (is_home_zone, is_work_zone, is_known_area, is_unfamiliar_area) before any scoring. Only the boolean flags and computed risk scores are persisted. This ensures user location privacy even if data is compromised."),

        calloutBox("30-Second Critical Persistence", "Before triggering emergency contacts at the critical risk level, the system requires 30 seconds of sustained critical risk. This prevents single-frame spikes from triggering irreversible actions while still responding quickly to genuine emergencies."),
        spacer(),

        // ═══════════════════════════════════════════════════════
        // 17. REFERENCES
        // ═══════════════════════════════════════════════════════
        heading1("17. Scientific References"),

        para("Tanaka H, Monahan KD, Seals DR (2001). Age-predicted maximal heart rate revisited. Journal of the American College of Cardiology, 37(1), 153-156.", { italic: true }),
        para("British Thoracic Society (2017). BTS Guideline for Oxygen Use in Adults in Healthcare and Emergency Settings. Thorax, 72(Suppl 1), ii1-ii90.", { italic: true }),
        para("World Health Organization (2022). Pulse Oximetry Training Manual. WHO Press.", { italic: true }),
        para("Esco MR, Flatt AA (2014). Ultra-short-term heart rate variability indexes at rest and post-exercise in athletes. Journal of Sports Science and Medicine, 13(3), 535-541.", { italic: true }),
        para("Shaffer F, Ginsberg JP (2017). An overview of heart rate variability metrics and norms. Frontiers in Public Health, 5, 258.", { italic: true }),
        para("Thayer JF, Ahs F, Fredrikson M, Sollers JJ, Wager TD (2012). A meta-analysis of heart rate variability and neuroimaging studies. Neuroscience and Biobehavioral Reviews, 36(2), 747-756.", { italic: true }),
        para("Vila J, et al. (2007). Cardiac defense: From attention to action. International Journal of Psychophysiology, 66(3), 169-182.", { italic: true }),

        spacer(200),
        divider(),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 }, children: [new TextRun({ text: "\u2014 End of Document \u2014", font: "Arial", size: 20, color: C.textLight })] }),
      ],
    },
  ],
});

// ── Save ──────────────────────────────────────────────────────
Packer.toBuffer(doc).then(buffer => {
  const outPath = "/Users/rithvik/Desktop/WUALT_D2C/SCIENTIFIC_PROCESS_DOCUMENT.docx";
  fs.writeFileSync(outPath, buffer);
  console.log(`Saved: ${outPath} (${(buffer.length / 1024).toFixed(0)} KB)`);
});
