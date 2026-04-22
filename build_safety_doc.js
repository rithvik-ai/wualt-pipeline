const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat, TabStopType, TabStopPosition,
  ExternalHyperlink,
} = require("docx");

// ── Colours ──────────────────────────────────────────────────
const C = {
  primary:   "1B2A4A",
  accent:    "2E75B6",
  accent2:   "4A90D9",
  green:     "2D8A4E",
  amber:     "C07D10",
  red:       "C0392B",
  purple:    "7C3AED",
  orange:    "EA580C",
  headerBg:  "1B2A4A",
  headerTxt: "FFFFFF",
  rowAlt:    "F0F4FA",
  rowWhite:  "FFFFFF",
  lightBlue: "E8F0FE",
  lightGrey: "F5F6F8",
  lightPurple:"F3F0FF",
  lightRed:  "FEF2F2",
  lightGreen:"F0FDF4",
  lightAmber:"FFFBEB",
  border:    "B0B8C8",
  divider:   "2E75B6",
  textDark:  "1A1A2E",
  textMid:   "3A3A5C",
  textLight: "5A5A7A",
};

// ── Page dims (US Letter) ────────────────────────────────────
const PAGE_W = 12240;
const PAGE_H = 15840;
const MARGIN = 1440;
const CONTENT_W = PAGE_W - 2 * MARGIN; // 9360

// ── Borders ──────────────────────────────────────────────────
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: C.border };
const borders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };
const noBorders = {
  top: { style: BorderStyle.NONE, size: 0 },
  bottom: { style: BorderStyle.NONE, size: 0 },
  left: { style: BorderStyle.NONE, size: 0 },
  right: { style: BorderStyle.NONE, size: 0 },
};

// ── Helpers ──────────────────────────────────────────────────
function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, children: [new TextRun(text)] });
}

function para(text, opts = {}) {
  const runs = [];
  if (typeof text === "string") {
    runs.push(new TextRun({ text, size: opts.size || 22, font: "Arial", color: opts.color || C.textDark, bold: opts.bold, italics: opts.italics }));
  } else {
    text.forEach(t => runs.push(new TextRun({ size: 22, font: "Arial", color: C.textDark, ...t })));
  }
  return new Paragraph({
    children: runs,
    spacing: { after: opts.after !== undefined ? opts.after : 120, before: opts.before || 0, line: opts.line || 276 },
    alignment: opts.align || AlignmentType.LEFT,
  });
}

function bulletItem(text, opts = {}) {
  const runs = typeof text === "string"
    ? [new TextRun({ text, size: 22, font: "Arial", color: C.textDark })]
    : text.map(t => new TextRun({ size: 22, font: "Arial", color: C.textDark, ...t }));
  return new Paragraph({
    numbering: { reference: "bullets", level: opts.level || 0 },
    children: runs,
    spacing: { after: 80, line: 276 },
  });
}

function numberItem(text, ref = "numbers", level = 0) {
  const runs = typeof text === "string"
    ? [new TextRun({ text, size: 22, font: "Arial", color: C.textDark })]
    : text.map(t => new TextRun({ size: 22, font: "Arial", color: C.textDark, ...t }));
  return new Paragraph({
    numbering: { reference: ref, level },
    children: runs,
    spacing: { after: 80, line: 276 },
  });
}

function calloutBox(title, body, bgColor, accentColor) {
  const cellW = CONTENT_W;
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [cellW],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: {
              top: { style: BorderStyle.SINGLE, size: 1, color: accentColor },
              bottom: { style: BorderStyle.SINGLE, size: 1, color: accentColor },
              left: { style: BorderStyle.SINGLE, size: 12, color: accentColor },
              right: { style: BorderStyle.SINGLE, size: 1, color: accentColor },
            },
            shading: { fill: bgColor, type: ShadingType.CLEAR },
            width: { size: cellW, type: WidthType.DXA },
            margins: { top: 120, bottom: 120, left: 200, right: 200 },
            children: [
              new Paragraph({
                children: [new TextRun({ text: title, bold: true, size: 22, font: "Arial", color: accentColor })],
                spacing: { after: 60 },
              }),
              new Paragraph({
                children: [new TextRun({ text: body, size: 20, font: "Arial", color: C.textMid })],
                spacing: { after: 0, line: 260 },
              }),
            ],
          }),
        ],
      }),
    ],
  });
}

function makeTable(headers, rows, colWidths) {
  const headerRow = new TableRow({
    children: headers.map((h, i) =>
      new TableCell({
        borders,
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: { fill: C.headerBg, type: ShadingType.CLEAR },
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({
          children: [new TextRun({ text: h, bold: true, size: 20, font: "Arial", color: C.headerTxt })],
          alignment: AlignmentType.LEFT,
        })],
      })
    ),
  });

  const dataRows = rows.map((row, ri) =>
    new TableRow({
      children: row.map((cell, ci) =>
        new TableCell({
          borders,
          width: { size: colWidths[ci], type: WidthType.DXA },
          shading: { fill: ri % 2 === 0 ? C.rowWhite : C.rowAlt, type: ShadingType.CLEAR },
          margins: { top: 50, bottom: 50, left: 100, right: 100 },
          children: [new Paragraph({
            children: typeof cell === "string"
              ? [new TextRun({ text: cell, size: 20, font: "Arial", color: C.textDark })]
              : cell.map(t => new TextRun({ size: 20, font: "Arial", color: C.textDark, ...t })),
            spacing: { after: 0, line: 260 },
          })],
        })
      ),
    })
  );

  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

function spacer(pts = 200) {
  return new Paragraph({ spacing: { after: pts } });
}

// ══════════════════════════════════════════════════════════════
//  BUILD DOCUMENT
// ══════════════════════════════════════════════════════════════

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22 } },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: C.primary },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: C.accent },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: C.textDark },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
        ],
      },
      {
        reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        ],
      },
      {
        reference: "numbers2",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        ],
      },
      {
        reference: "numbers3",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_W, height: PAGE_H },
          margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              children: [
                new TextRun({ text: "WUALT \u2014 Geospatial Safety Risk Model", size: 16, font: "Arial", color: C.textLight }),
                new TextRun({ text: "\tConfidential", size: 16, font: "Arial", color: C.textLight }),
              ],
              tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
              border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: C.accent, space: 8 } },
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              children: [
                new TextRun({ text: "WUALT Safety Intelligence Engine", size: 16, font: "Arial", color: C.textLight }),
                new TextRun({ text: "\tPage ", size: 16, font: "Arial", color: C.textLight }),
                new TextRun({ children: [PageNumber.CURRENT], size: 16, font: "Arial", color: C.textLight }),
              ],
              tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
              border: { top: { style: BorderStyle.SINGLE, size: 4, color: C.accent, space: 8 } },
            }),
          ],
        }),
      },
      children: [

        // ════════════════════════════════════════════════════════
        // TITLE PAGE
        // ════════════════════════════════════════════════════════
        spacer(1200),
        para("WUALT", { size: 20, color: C.accent, bold: true, align: AlignmentType.CENTER, after: 60 }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 120 },
          children: [new TextRun({ text: "Geospatial Context &", size: 48, bold: true, font: "Arial", color: C.primary })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "Safety Risk Model", size: 48, bold: true, font: "Arial", color: C.accent })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 60 },
          border: { top: { style: BorderStyle.SINGLE, size: 6, color: C.accent, space: 12 } },
          children: [],
        }),
        para("Technical Architecture, Rule Engine Logic, Risk Scoring System,\nand Implementation Guide", { size: 22, color: C.textMid, align: AlignmentType.CENTER, after: 400 }),

        makeTable(
          ["Property", "Value"],
          [
            ["Document Version", "2.0"],
            ["Date", "April 2026"],
            ["Classification", "Confidential \u2014 Internal Engineering"],
            ["System", "WUALT Smart Ring \u2014 Safety Intelligence Engine"],
            ["Architecture", "Rule-Based / Interpretable (No ML)"],
            ["Latency Target", "< 700 ms per evaluation"],
            ["Risk Levels", "5 (normal \u2192 low \u2192 moderate \u2192 high \u2192 critical)"],
            ["Detection Layers", "3 (Physiological + Fall + Geospatial)"],
          ],
          [3200, 6160]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 1. EXECUTIVE SUMMARY
        // ════════════════════════════════════════════════════════
        heading("1. Executive Summary"),
        para("The WUALT Geospatial Safety Risk Model extends the existing physiological distress detection engine by incorporating location, movement, time, and familiarity context. The system answers a single question:"),
        calloutBox(
          "Core Question",
          "\"Given the user's body signals and current context, how concerning is the situation right now?\"",
          C.lightBlue, C.accent
        ),
        spacer(100),
        para("The model combines two independent signal streams \u2014 physiological state (heart rate, SpO2, HRV, temperature) and geospatial context (GPS location, time of day, movement patterns, zone familiarity) \u2014 to produce a unified 5-level safety risk classification with explainable reasoning and calm, actionable alerts."),
        para("Key design principles:"),
        bulletItem("Context is a modifier, never a trigger \u2014 geospatial data amplifies or suppresses physiological signals"),
        bulletItem("Every decision is traceable \u2014 the reasoning chain shows exactly why a risk level was assigned"),
        bulletItem("Privacy-first \u2014 raw GPS coordinates are never stored, only zone classifications"),
        bulletItem("Women\u2019s safety focus \u2014 escalation patterns tuned for real-world threat scenarios"),
        bulletItem("Low false positives \u2014 suppression rules prevent alerts during exercise, at home, or at work"),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 2. SYSTEM ARCHITECTURE
        // ════════════════════════════════════════════════════════
        heading("2. System Architecture"),
        para("The unified safety engine operates as a three-layer detection stack:"),

        heading("2.1 Layer 1: Physiological Distress Detection", HeadingLevel.HEADING_2),
        para("The existing distress engine processes raw wearable sensor data through a seven-stage pipeline: SQI gate, motion classification, signal flagging (z-score + absolute thresholds), motion-aware suppression, state determination, persistence tracking, and alert selection."),
        para("Outputs one of three states: normal, stress, or distress \u2014 with a confidence score (0\u20131) and a list of contributing signals."),

        heading("2.2 Layer 2: Fall Detection", HeadingLevel.HEADING_2),
        para("A three-stage accelerometer-based fall detector:"),
        bulletItem([{ text: "Stage 1 \u2014 Impact Detection: ", bold: true }, { text: "Free-fall (acc_mag < 0.5g) followed by impact spike (> 3.0g) within a 5-second window" }]),
        bulletItem([{ text: "Stage 2 \u2014 Orientation Change: ", bold: true }, { text: "Body angle shift > 60\u00B0 from pre-fall vertical, estimated from accelerometer axis ratios" }]),
        bulletItem([{ text: "Stage 3 \u2014 Post-Fall Inactivity: ", bold: true }, { text: "Stillness (dyn_acc < 0.05g) sustained for 3+ seconds confirms the fall; movement resumes \u2192 false alarm" }]),

        heading("2.3 Layer 3: Geospatial Safety Risk (New)", HeadingLevel.HEADING_2),
        para("The geospatial layer takes the physiological output and combines it with context to produce a unified risk assessment. It does NOT detect anything independently \u2014 it modifies the physiological signal based on how concerning the current context is."),
        para("Architecture flow:"),
        numberItem("Score each context dimension independently (time, location, movement, connectivity, familiarity)"),
        numberItem("Compute weighted context score (0\u20131)"),
        numberItem("Multiply physiological base risk by context amplifier (0.8x\u20131.5x)"),
        numberItem("Apply suppression rules for safe contexts (home, gym, work)"),
        numberItem("Apply escalation patterns for dangerous combinations"),
        numberItem("Map final score to 5-level risk classification"),
        numberItem("Select context-appropriate alert message"),

        heading("2.4 Unified Engine (Orchestrator)", HeadingLevel.HEADING_2),
        para("The UnifiedSafetyEngine class orchestrates all three layers in a single evaluate() call. It runs physiological detection first, then fall detection, then geospatial risk assessment, and finally selects the highest-priority alert across all layers."),

        calloutBox(
          "Design Principle",
          "The geospatial context NEVER triggers alerts by itself. Even at 2 AM in an unfamiliar area with no phone, if the user\u2019s body signals are normal, the risk stays low. Context only matters when the body is also showing concern.",
          C.lightPurple, C.purple
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 3. INPUT SPECIFICATION
        // ════════════════════════════════════════════════════════
        heading("3. Input Specification"),

        heading("3.1 Physiological Inputs (from Distress Engine)", HeadingLevel.HEADING_2),
        makeTable(
          ["Field", "Type", "Description"],
          [
            ["state", "string", "\"normal\" | \"stress\" | \"distress\""],
            ["confidence", "float", "Detection confidence (0.0 \u2013 1.0)"],
            ["contributing_signals", "list", "Which signals triggered: hr, hrv, spo2, temp"],
          ],
          [2000, 1500, 5860]
        ),
        spacer(120),

        heading("3.2 Geospatial Context Inputs (from Phone GPS)", HeadingLevel.HEADING_2),
        para("All fields have safe defaults so the system gracefully degrades when partial context is available."),
        spacer(60),
        makeTable(
          ["Field", "Type", "Default", "Description"],
          [
            ["latitude", "float", "0.0", "GPS latitude (used for distance calc only, never stored)"],
            ["longitude", "float", "0.0", "GPS longitude (used for distance calc only, never stored)"],
            ["timestamp", "int", "0", "Unix timestamp of the reading"],
            ["speed_kmph", "float", "0.0", "Current speed in km/h"],
            ["heading", "float", "0.0", "Direction of travel in degrees"],
            ["is_home_zone", "bool", "false", "User is within their defined home geofence"],
            ["is_work_zone", "bool", "false", "User is within their defined work geofence"],
            ["is_known_area", "bool", "true", "Area has been visited before (default safe)"],
            ["is_unfamiliar_area", "bool", "false", "Area has never been visited"],
            ["distance_from_home_km", "float", "0.0", "Straight-line distance from home"],
            ["hour_of_day", "int", "12", "Current hour (0\u201323)"],
            ["is_night", "bool", "false", "Whether it is nighttime (20:00\u201305:59)"],
            ["is_stationary", "bool", "true", "User is not moving"],
            ["is_walking", "bool", "false", "User is walking (GPS speed 2\u20137 km/h)"],
            ["is_vehicle_like_motion", "bool", "false", "Speed suggests vehicle (> 15 km/h)"],
            ["sudden_route_change", "bool", "false", "Heading changed > 90\u00B0 unexpectedly"],
            ["sudden_stop", "bool", "false", "Speed dropped from > 20 to < 2 km/h rapidly"],
            ["phone_connected", "bool", "true", "Ring has active BLE connection to phone"],
          ],
          [2200, 900, 900, 5360]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 4. OUTPUT SPECIFICATION
        // ════════════════════════════════════════════════════════
        heading("4. Output Specification"),
        para("The unified engine returns a complete safety assessment:"),
        makeTable(
          ["Field", "Type", "Description"],
          [
            ["risk_level", "string", "\"normal\" | \"low_risk\" | \"moderate_risk\" | \"high_risk\" | \"critical\""],
            ["risk_score", "float", "Combined risk score (0.0 \u2013 1.0)"],
            ["reasoning", "list[str]", "Ordered list of factors that contributed to the risk level"],
            ["recommended_action", "string", "What the system recommends doing at this risk level"],
            ["alert.title", "string", "Short, calm alert title for the user"],
            ["alert.message", "string", "Human-friendly, actionable alert message"],
            ["alert.severity", "string", "\"low\" | \"medium\" | \"high\""],
            ["debug.physio_state", "string", "The underlying physiological state"],
            ["debug.physio_confidence", "float", "Physiological detection confidence"],
            ["debug.context_score", "float", "Raw geospatial context risk score"],
            ["debug.geo_available", "bool", "Whether geospatial context was provided"],
          ],
          [2800, 1500, 5060]
        ),
        spacer(120),

        heading("4.1 Risk Level Definitions", HeadingLevel.HEADING_2),
        makeTable(
          ["Level", "Score Range", "Meaning", "System Response"],
          [
            ["NORMAL", "< 0.20", "No concern. Safe context.", "Continue passive monitoring"],
            ["LOW RISK", "0.20 \u2013 0.39", "Mild stress or slightly unusual context", "Log context for pattern analysis"],
            ["MODERATE", "0.40 \u2013 0.59", "Stress in concerning context, or distress in safe context", "Suggest user check in; enable emergency contacts"],
            ["HIGH RISK", "0.60 \u2013 0.79", "Distress in concerning context", "Prompt user to confirm safety; prepare to notify contacts"],
            ["CRITICAL", "\u2265 0.80", "Multiple danger signals across physiological and context", "Initiate safety protocol; auto-notify if no response in 60s"],
          ],
          [1600, 1400, 3360, 3000]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 5. RISK SCORING SYSTEM
        // ════════════════════════════════════════════════════════
        heading("5. Risk Scoring System"),

        heading("5.1 Core Formula", HeadingLevel.HEADING_2),
        calloutBox(
          "Risk Score Formula",
          "risk_score = physio_base \u00D7 max(0.3, confidence) \u00D7 context_amplifier\n\nWhere context_amplifier = 0.8 + (context_score \u00D7 0.7)\n\nRange: 0.8x (safe context) to 1.5x (dangerous context)",
          C.lightBlue, C.accent
        ),
        spacer(120),

        heading("5.2 Physiological Base Risk", HeadingLevel.HEADING_2),
        makeTable(
          ["Physio State", "Base Score", "Rationale"],
          [
            ["Normal", "0.05", "Very low base \u2014 body is fine, context alone shouldn\u2019t alarm"],
            ["Stress", "0.35", "Moderate base \u2014 body is showing concern, context determines severity"],
            ["Distress", "0.70", "High base \u2014 body is in distress, context can escalate to critical"],
          ],
          [2000, 1500, 5860]
        ),
        spacer(100),
        para("The base score is multiplied by max(0.3, confidence) to prevent very low-confidence detections from producing high risk scores. Even if the distress engine outputs \"distress\" with 0.1 confidence, the effective base is 0.70 \u00D7 0.3 = 0.21 rather than 0.07."),

        heading("5.3 Context Dimensions and Weights", HeadingLevel.HEADING_2),
        para("The context score is a weighted combination of five independent dimensions, each scored 0\u20131:"),
        makeTable(
          ["Dimension", "Weight", "What It Measures", "Key Thresholds"],
          [
            ["Time of Day", "25%", "How risky the current hour is", "Late night (23\u201304): 0.9 | Night (20\u201322): 0.6 | Dawn (05\u201306): 0.4 | Day: 0.1"],
            ["Location", "30%", "Zone familiarity + distance from home", "Unfamiliar: 0.7 | Known: 0.15 | Work: 0.05 | Home: 0.0 | Distance adds 0\u20130.24"],
            ["Movement", "25%", "Movement anomalies and patterns", "Route change: +0.5 | Sudden stop: +0.4 | Vehicle at night: +0.2 | High speed: +0.3"],
            ["Connectivity", "10%", "Phone connection status", "Disconnected: 0.8 | Connected: 0.0"],
            ["Familiarity", "10%", "Secondary unfamiliarity check", "Unfamiliar + not known: 0.6 | Otherwise: 0.0"],
          ],
          [1600, 1000, 2760, 4000]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 6. TIME RISK SCORING
        // ════════════════════════════════════════════════════════
        heading("6. Time Risk Scoring"),
        para("Time of day is one of the strongest contextual signals. Crime statistics and safety research consistently show that late-night hours carry disproportionate risk, particularly for women\u2019s safety scenarios."),
        makeTable(
          ["Time Window", "Hours", "Score", "Rationale"],
          [
            ["Late Night", "23:00 \u2013 04:59", "0.9", "Peak risk hours. Fewer people around, reduced visibility, limited help access"],
            ["Night", "20:00 \u2013 22:59", "0.6", "Elevated risk. Darkness, reduced foot traffic"],
            ["Early Morning", "05:00 \u2013 06:59", "0.4", "Transitional period. Dawn, some risk remains"],
            ["Daytime", "07:00 \u2013 19:59", "0.1", "Lowest risk. Public activity, visibility, help readily available"],
          ],
          [1800, 2000, 1000, 4560]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 7. LOCATION RISK SCORING
        // ════════════════════════════════════════════════════════
        heading("7. Location & Familiarity Scoring"),

        heading("7.1 Zone Classification", HeadingLevel.HEADING_2),
        makeTable(
          ["Zone", "Score", "How It\u2019s Determined"],
          [
            ["Home Zone", "0.0", "User-defined geofence around home address (typically 100\u2013500m radius)"],
            ["Work Zone", "0.05", "User-defined geofence around workplace"],
            ["Known Area", "0.15", "Area visited 3+ times in past 30 days (automatically learned)"],
            ["Unfamiliar Area", "0.70", "No prior visits detected. First-time location."],
          ],
          [2000, 1000, 6360]
        ),
        spacer(100),

        heading("7.2 Distance Risk Tiers", HeadingLevel.HEADING_2),
        para("Distance from home adds an incremental risk contribution on top of zone classification:"),
        makeTable(
          ["Distance from Home", "Additive Score", "Effective Contribution"],
          [
            ["\u2265 50 km", "+0.24 (0.8 \u00D7 0.3)", "Far from home, unfamiliar territory"],
            ["\u2265 20 km", "+0.18 (0.6 \u00D7 0.3)", "Significantly away from home"],
            ["\u2265 5 km", "+0.09 (0.3 \u00D7 0.3)", "Moderately away from home"],
            ["< 5 km", "+0.0", "Close to home, minimal distance risk"],
          ],
          [2600, 2400, 4360]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 8. MOVEMENT RISK SCORING
        // ════════════════════════════════════════════════════════
        heading("8. Movement Risk Scoring"),
        para("Movement anomalies are strong indicators of concerning situations. These signals are additive \u2014 multiple anomalies compound:"),
        makeTable(
          ["Signal", "Score", "Detection Method", "Safety Rationale"],
          [
            ["Sudden Route Change", "+0.5", "Heading changed > 90\u00B0 from expected path", "Forced detour, abduction scenario, getting lost"],
            ["Sudden Stop", "+0.4", "Speed dropped from > 20 to < 2 km/h rapidly", "Vehicle forced to stop, breakdown in unsafe area"],
            ["Vehicle at Night", "+0.2", "Vehicle-like speed (> 15 km/h) during night hours", "Higher risk of road incidents, unfamiliar drivers"],
            ["High Speed", "+0.3", "Speed exceeds 120 km/h", "Erratic/reckless driving, potential danger"],
          ],
          [1800, 900, 2860, 3800]
        ),
        spacer(100),
        para([{ text: "Maximum movement score: ", bold: true }, { text: "1.0 (capped). A sudden route change + sudden stop already scores 0.9." }]),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 9. SUPPRESSION RULES
        // ════════════════════════════════════════════════════════
        heading("9. Suppression Rules (False Positive Reduction)"),
        para("Suppression rules reduce false positives by recognizing safe contexts where elevated physiological signals are expected or non-threatening. These rules ONLY apply when physio_state is \"stress\" \u2014 distress is never suppressed."),
        spacer(60),

        calloutBox(
          "Important",
          "Suppression NEVER applies to distress. If the body shows multi-signal distress, it always receives at least moderate risk regardless of context.",
          C.lightRed, C.red
        ),
        spacer(120),

        makeTable(
          ["Rule", "Multiplier", "Conditions", "Example Scenario"],
          [
            ["Home Safe", "\u00D70.5", "Home zone + daytime + no movement anomaly", "Watching a thriller at home, HR spikes \u2192 risk halved"],
            ["Work Safe", "\u00D70.6", "Work zone + business hours (07\u201320) + no sudden stop", "Stressful meeting at the office \u2192 risk reduced 40%"],
            ["Known Active", "\u00D70.7", "Known area + walking + daytime + no route change", "Working out at the gym, HR elevated \u2192 risk reduced 30%"],
          ],
          [1600, 1200, 3360, 3200]
        ),
        spacer(100),

        para("Example walkthrough:"),
        bulletItem("User is at the gym (known area, walking, daytime). HR = 110 bpm \u2192 stress state."),
        bulletItem("Base risk: 0.35 \u00D7 0.45 (confidence) = 0.158"),
        bulletItem("Context amplifier: ~0.84 (low context risk) \u2192 0.158 \u00D7 0.84 = 0.133"),
        bulletItem("Suppression: \u00D70.7 \u2192 0.133 \u00D7 0.7 = 0.093"),
        bulletItem("Final risk: 0.093 \u2192 NORMAL. No alert shown."),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 10. ESCALATION PATTERNS
        // ════════════════════════════════════════════════════════
        heading("10. Escalation Patterns"),
        para("Escalation patterns detect specific dangerous combinations that deserve higher risk than the formula alone would produce. These force a minimum risk score regardless of the computed value."),
        spacer(60),

        heading("10.1 Critical Patterns (\u2265 0.85)", HeadingLevel.HEADING_2),
        makeTable(
          ["Pattern", "Min Score", "Signals Required"],
          [
            ["Distress + Night + Unfamiliar + No Phone", "0.90", "Physiological distress in an unknown area at night with lost phone connection"],
            ["Distress + Sudden Stop + Unfamiliar", "0.85", "Body in distress, vehicle suddenly stopped in unfamiliar territory"],
          ],
          [3500, 1200, 4660]
        ),
        spacer(100),

        heading("10.2 High Risk Patterns (\u2265 0.65)", HeadingLevel.HEADING_2),
        makeTable(
          ["Pattern", "Min Score", "Signals Required"],
          [
            ["Distress + Night + Unfamiliar", "0.75", "Distress at night in an area never visited before"],
            ["Distress + Route Deviation", "0.70", "Distress combined with unexpected route change"],
            ["Vehicle Sudden Stop + Physio Concern", "0.65", "Stress or distress with vehicle coming to sudden halt"],
          ],
          [3500, 1200, 4660]
        ),
        spacer(100),

        heading("10.3 Elevated Patterns (\u2265 0.50)", HeadingLevel.HEADING_2),
        makeTable(
          ["Pattern", "Min Score", "Signals Required"],
          [
            ["Stress + Night + Unfamiliar + Far", "0.55", "Stress at night, unfamiliar, > 10 km from home"],
            ["Phone Lost + Unfamiliar + Night", "0.50", "Phone disconnected in unknown area at night"],
          ],
          [3500, 1200, 4660]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 11. ALERT SYSTEM
        // ════════════════════════════════════════════════════════
        heading("11. Alert System"),
        para("All alerts follow the STAR principle \u2014 calm, clear, actionable, and privacy-respecting. The system never uses medical jargon, technical terms, or alarming language."),
        spacer(60),

        heading("11.1 Alert Design Principles", HeadingLevel.HEADING_2),
        makeTable(
          ["Principle", "Good Example", "Bad Example"],
          [
            ["Human language", "\"We noticed signs of stress in an unfamiliar setting.\"", "\"Geospatial anomaly detected in Zone 4.\""],
            ["Actionable", "\"Consider contacting someone you trust.\"", "\"Risk level elevated to HIGH.\""],
            ["Calm tone", "\"Are you okay?\"", "\"DANGER: Multiple warning signals!\""],
            ["Privacy-respecting", "\"Your surroundings seem unusual.\"", "\"You are at 17.385\u00B0N, 78.486\u00B0E, an unfamiliar location.\""],
            ["Non-judgmental", "\"We\u2019re here if you need us.\"", "\"You should not be in this area at this time.\""],
          ],
          [2000, 3680, 3680]
        ),
        spacer(120),

        heading("11.2 Alert Examples by Risk Level", HeadingLevel.HEADING_2),
        makeTable(
          ["Risk Level", "Alert Title", "Alert Message"],
          [
            ["NORMAL", "All clear", "You seem to be in a familiar place and everything looks fine. We\u2019re here if you need us."],
            ["LOW RISK", "Keeping an eye out", "Things look mostly fine, but we noticed a small change. We\u2019ll keep monitoring quietly."],
            ["MODERATE", "Stay aware", "We\u2019re picking up some signs worth noting. If you feel uneasy, consider reaching out to someone you trust."],
            ["HIGH RISK", "We\u2019re concerned", "We noticed signs of distress in an unfamiliar setting. Are you okay? Consider contacting someone you trust."],
            ["CRITICAL", "Emergency \u2014 are you safe?", "Multiple warning signs detected. If you\u2019re in danger, please call for help. We can alert your trusted contacts."],
          ],
          [1600, 2200, 5560]
        ),
        spacer(100),

        heading("11.3 Context-Specific Alerts", HeadingLevel.HEADING_2),
        para("The system picks specialized alerts for specific patterns:"),
        bulletItem([{ text: "Route deviation: ", bold: true }, { text: "\"We noticed you\u2019ve deviated from your usual path. If everything\u2019s fine, no action needed \u2014 we\u2019re just keeping watch.\"" }]),
        bulletItem([{ text: "Phone disconnected: ", bold: true }, { text: "\"We\u2019ve lost connection to your phone. If you\u2019re in an unfamiliar area, try to stay in well-lit, public spaces.\"" }]),
        bulletItem([{ text: "Fall confirmed: ", bold: true }, { text: "\"A fall has been detected and you haven\u2019t moved. If you need help, please call out or press the alert button.\"" }]),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 12. EXAMPLE SCENARIOS
        // ════════════════════════════════════════════════════════
        heading("12. Example Scenarios"),

        heading("12.1 Normal at Home \u2014 Daytime", HeadingLevel.HEADING_2),
        makeTable(
          ["Input", "Value"],
          [
            ["Physiological", "HR=72, SpO2=98.5%, state=NORMAL"],
            ["Context", "Home zone, 14:00, phone connected, stationary"],
            ["Context Score", "0.038"],
            ["Risk Score", "0.038"],
            ["Risk Level", "NORMAL"],
            ["Alert", "\"You seem to be in a familiar place and everything looks fine.\""],
          ],
          [2200, 7160]
        ),
        spacer(80),

        heading("12.2 Stressed at Gym (Suppressed)", HeadingLevel.HEADING_2),
        makeTable(
          ["Input", "Value"],
          [
            ["Physiological", "HR=110, SpO2=97%, state=STRESS"],
            ["Context", "Known area, 10:00, walking, phone connected, 1.5 km from home"],
            ["Suppression", "Known area + walking + daytime \u2192 \u00D70.7"],
            ["Risk Score", "0.097"],
            ["Risk Level", "NORMAL (suppressed from low_risk)"],
            ["Reasoning", "stress state detected \u2192 suppressed: known area, walking, daytime"],
          ],
          [2200, 7160]
        ),
        spacer(80),

        heading("12.3 Distress at Night in Unfamiliar Area", HeadingLevel.HEADING_2),
        makeTable(
          ["Input", "Value"],
          [
            ["Physiological", "HR=130, SpO2=91%, state=DISTRESS, signals=[hr, spo2]"],
            ["Context", "Unfamiliar area, 23:00, stationary, 15 km from home, phone connected"],
            ["Escalation", "Distress + night + unfamiliar \u2192 force 0.75"],
            ["Risk Score", "0.75"],
            ["Risk Level", "HIGH RISK"],
            ["Alert", "\"We noticed signs of distress in an unfamiliar setting. Are you okay?\""],
            ["Action", "Prompt user to confirm safety. Prepare to notify trusted contacts."],
          ],
          [2200, 7160]
        ),
        spacer(80),

        heading("12.4 Vehicle Sudden Stop at Night", HeadingLevel.HEADING_2),
        makeTable(
          ["Input", "Value"],
          [
            ["Physiological", "HR=125, SpO2=93%, state=DISTRESS, signals=[hr, spo2]"],
            ["Context", "Unfamiliar, 01:00, vehicle, sudden stop + route change, 25 km from home"],
            ["Escalation", "Distress + sudden stop + unfamiliar \u2192 force 0.85"],
            ["Risk Score", "0.85"],
            ["Risk Level", "HIGH RISK (would escalate to CRITICAL after 30s persistence)"],
            ["Reasoning", "distress + late night + unfamiliar + route deviation + sudden stop + vehicle"],
          ],
          [2200, 7160]
        ),
        spacer(80),

        heading("12.5 Critical \u2014 Worst Case", HeadingLevel.HEADING_2),
        makeTable(
          ["Input", "Value"],
          [
            ["Physiological", "HR=140, SpO2=89%, state=DISTRESS, signals=[hr, spo2]"],
            ["Context", "Unfamiliar, 02:00, phone disconnected, 50 km from home, stationary"],
            ["Escalation", "Distress + night + unfamiliar + no phone \u2192 force 0.90"],
            ["Risk Score", "0.90"],
            ["Risk Level", "HIGH RISK \u2192 CRITICAL (after 30s persistence)"],
            ["Alert", "\"Multiple warning signs detected. If you\u2019re in danger, please call for help.\""],
            ["Action", "Initiate safety protocol. Notify emergency contacts if no response within 60s."],
          ],
          [2200, 7160]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 13. ANTI-FLICKERING
        // ════════════════════════════════════════════════════════
        heading("13. Anti-Flickering & Persistence"),
        para("To prevent rapid oscillation between risk levels (which would cause repeated alerts and user fatigue), the system implements persistence requirements:"),
        bulletItem([{ text: "Critical persistence: ", bold: true }, { text: "The risk must stay at critical level for 30 consecutive seconds before the system outputs \"critical\". Until then, it caps at \"high_risk\"." }]),
        bulletItem([{ text: "Escalation cooldown: ", bold: true }, { text: "Once the high-risk timer resets (risk drops below high), it starts fresh on the next occurrence." }]),
        bulletItem([{ text: "History tracking: ", bold: true }, { text: "The engine maintains a 120-frame rolling history (~2 minutes at 1 Hz) for state distribution analysis." }]),

        calloutBox(
          "Why 30 Seconds?",
          "30 seconds is long enough to filter out transient GPS glitches, momentary sensor noise, or brief stops at traffic lights \u2014 but short enough that a real emergency still triggers quickly. This value is configurable.",
          C.lightAmber, C.amber
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 14. PRIVACY DESIGN
        // ════════════════════════════════════════════════════════
        heading("14. Privacy-First Design"),
        para("Privacy is a core architectural constraint, not an afterthought. For a women\u2019s safety product, the device must protect without surveilling."),
        spacer(60),

        heading("14.1 Data Flow", HeadingLevel.HEADING_2),
        makeTable(
          ["Data", "Stored?", "Transmitted?", "Purpose"],
          [
            ["Raw GPS (lat/lon)", "NEVER", "Never", "Used once for distance calc, then discarded"],
            ["Zone classification", "Yes (local)", "Only in alerts", "is_home, is_work, is_known, is_unfamiliar"],
            ["Distance from home", "Session only", "Never", "Computed per-frame, not persisted"],
            ["Time of day", "Session only", "Never", "Used for time risk scoring"],
            ["Movement flags", "Session only", "Only in alerts", "sudden_stop, route_change booleans"],
            ["Risk level", "Yes (local)", "In alerts only", "For history and pattern analysis"],
            ["Reasoning chain", "Session only", "In alerts only", "For user transparency"],
          ],
          [2200, 1200, 1400, 4560]
        ),
        spacer(100),

        heading("14.2 Privacy Principles", HeadingLevel.HEADING_2),
        bulletItem("Raw coordinates are processed in-memory and immediately discarded after zone classification"),
        bulletItem("No location history is built or stored \u2014 each frame is independent"),
        bulletItem("Alert messages never include coordinates, addresses, or identifiable locations"),
        bulletItem("Zone learning (\"known area\" detection) uses hashed grid cells, not raw GPS"),
        bulletItem("All processing happens on-device \u2014 no cloud dependency for risk assessment"),
        bulletItem("The user can disable geospatial context entirely; the system falls back to physio-only assessment"),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 15. TUNING STRATEGY
        // ════════════════════════════════════════════════════════
        heading("15. Threshold Tuning Strategy"),
        para("All thresholds are configurable constants at the top of the module. The recommended tuning approach:"),

        heading("15.1 Tuning Parameters", HeadingLevel.HEADING_2),
        makeTable(
          ["Parameter", "Current Value", "Tuning Guidance"],
          [
            ["GEO_WEIGHTS", "time:0.25, loc:0.30, move:0.25, conn:0.10, fam:0.10", "Increase movement weight if route deviations are strong signals in your user base"],
            ["TIME_RISK", "late_night:0.9, night:0.6, dawn:0.4, day:0.1", "Adjust based on local crime statistics and user feedback"],
            ["RISK_THRESHOLDS", "critical:0.80, high:0.60, mod:0.40, low:0.20", "Lower thresholds for more sensitive detection; raise for fewer false positives"],
            ["Suppression multipliers", "home:\u00D70.5, work:\u00D70.6, known:\u00D70.7", "Increase multipliers (closer to 1.0) for less suppression"],
            ["Critical persistence", "30 seconds", "Reduce for faster critical alerts; increase for fewer false criticals"],
          ],
          [2400, 2800, 4160]
        ),
        spacer(100),

        heading("15.2 Calibration Process", HeadingLevel.HEADING_2),
        numberItem("Collect 2\u20134 weeks of real-world data from beta users with geospatial context", "numbers2"),
        numberItem("Identify false positive scenarios (alerts in safe contexts) and false negatives (missed real concerns)", "numbers2"),
        numberItem("Adjust suppression multipliers to reduce false positives", "numbers2"),
        numberItem("Adjust escalation pattern thresholds to catch false negatives", "numbers2"),
        numberItem("A/B test with user cohorts to measure alert fatigue and perceived safety", "numbers2"),
        numberItem("Iterate monthly based on user feedback and incident reports", "numbers2"),

        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════
        // 16. IMPLEMENTATION REFERENCE
        // ════════════════════════════════════════════════════════
        heading("16. Implementation Reference"),

        heading("16.1 Key Classes", HeadingLevel.HEADING_2),
        makeTable(
          ["Class", "Purpose", "Location"],
          [
            ["GeoContext", "Typed dataclass for geospatial input with safe defaults", "distress_engine.py \u2014 Section 10"],
            ["SafetyRiskEngine", "Scores context, combines with physiology, applies suppression/escalation", "distress_engine.py \u2014 Section 11"],
            ["UnifiedSafetyEngine", "Top-level orchestrator combining all 3 detection layers", "distress_engine.py \u2014 Section 12"],
            ["DistressEngine", "Physiological distress detection (pre-existing)", "distress_engine.py \u2014 Section 7"],
            ["FallDetector", "3-stage fall detection state machine", "distress_engine.py \u2014 Section 5"],
          ],
          [2400, 4160, 2800]
        ),
        spacer(100),

        heading("16.2 Key Functions", HeadingLevel.HEADING_2),
        makeTable(
          ["Function", "Purpose"],
          [
            ["compute_context_score(ctx)", "Scores all 5 context dimensions, returns (score, reasons)"],
            ["_score_time_risk(ctx)", "Scores time-of-day risk (0\u20131)"],
            ["_score_location_risk(ctx)", "Scores zone familiarity + distance risk (0\u20131)"],
            ["_score_movement_risk(ctx)", "Scores movement anomalies (0\u20131)"],
            ["_compute_combined_risk()", "Combines physio base with context amplifier"],
            ["_apply_suppression()", "Reduces risk in safe contexts (home, gym, work)"],
            ["_apply_escalation()", "Forces minimum scores for dangerous pattern matches"],
          ],
          [3600, 5760]
        ),
        spacer(100),

        heading("16.3 Usage Example", HeadingLevel.HEADING_2),
        para("Python usage for real-time evaluation:", { after: 60 }),
        calloutBox(
          "Code Example",
          "engine = UnifiedSafetyEngine()\n\nfor frame in sensor_stream:\n    pipeline_out = preprocessor.step(frame)\n    geo_ctx = phone.get_context()  # from companion app\n    \n    result = engine.evaluate(pipeline_out, geo_context=geo_ctx)\n    \n    if result[\"safety\"][\"risk_level\"] in (\"high_risk\", \"critical\"):\n        show_alert(result[\"alert\"])\n        notify_contacts_if_needed(result[\"safety\"])",
          C.lightGrey, C.textMid
        ),

        spacer(400),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 200 },
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: C.accent, space: 12 } },
          children: [
            new TextRun({ text: "End of Document", size: 20, font: "Arial", color: C.textLight, italics: true }),
          ],
        }),
        para("WUALT Safety Intelligence Engine \u2014 Confidential", { size: 18, color: C.textLight, align: AlignmentType.CENTER }),

      ],
    },
  ],
});

// ── Write ────────────────────────────────────────────────────
const OUT = "/Users/rithvik/Desktop/WUALT_D2C/GEOSPATIAL_SAFETY_MODEL.docx";
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUT, buffer);
  console.log(`Saved: ${OUT} (${Math.round(buffer.length/1024)} KB)`);
});
