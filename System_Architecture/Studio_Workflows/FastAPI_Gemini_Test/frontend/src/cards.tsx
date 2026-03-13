/**
 * cards.tsx — ImpactStudio UI Card Design System Demo
 *
 * Shows the three card types the system will render:
 *   1. Chat response (conversation mode)
 *   2. Script Review card (original design from App.tsx)
 *   3. Impact Analysis card (new — community outreach advisory)
 *
 * Startup: make sure main.tsx renders <CardsDemo /> then run: npm run dev
 */

// No React state needed — all cards are static display components

const C = {
  bg: "#FAF7F2",
  bgWarm: "#F5F0E8",
  ink: "#1A1A1A",
  inkSoft: "#5C5549",
  inkMuted: "#9B9285",
  accent: "#C8553D",
  accentGlow: "#E86F56",
  gold: "#D4A03C",
  goldSoft: "#F2E6C4",
  teal: "#2A8F82",
  tealSoft: "#E6F5F2",
  tealMid: "#1D7268",
  cream: "#FFF9F0",
  card: "#FFFFFF",
  border: "#E8E2D8",
  borderLight: "#F0EBE3",
  shadow: "rgba(42,32,16,0.06)",
};

const font = "'Georgia','Times New Roman',serif";
const sans = "'Helvetica Neue','Arial',sans-serif";

// ─── Mock Data ──────────────────────────────────────────────────────────────

const MOCK_CHAT = {
  question: "Tell me what score you gave for FilmScoreTreatment.pdf?",
  answer:
    "Based on my earlier review of FilmScoreTreatment.pdf, I gave it a score of 4 out of 5. The screenplay showed strong uplifting value through Dave's inspiring transformation arc and the concept of the magical film score. The main areas I flagged for improvement were the abruptness of Connor's death and a few plot conveniences near the ending.",
};

const MOCK_REVIEW = {
  verdict: "Yes",
  score: 4,
  benefits: [
    "The core concept of a magical film score guiding life choices is unique and engaging.",
    "Dave's transformation from an introverted student to a successful individual is inspiring.",
    "The narrative explores themes of destiny, choice, and resilience in the face of tragedy.",
    "The story offers a hopeful resolution, emphasizing the enduring power of love and family.",
  ],
  risks: [
    "The sudden and devastating loss of Connor feels abrupt and may be too dark for an uplifting narrative.",
    "The mechanism of 'Film Score' failing due to external noise is somewhat contrived.",
    "Some plot points, like Rachel's near-fatal pre-eclampsia, might detract from the overall uplifting tone.",
  ],
  rationale:
    "This screenplay presents a compelling concept with a strong arc of personal growth and eventual triumph. Dave's journey from insecurity to success, guided by an external force, is inherently uplifting. While the narrative includes significant tragedy, the ultimate message of resilience, love, and second chances provides a hopeful conclusion.",
};

const MOCK_IMPACT = {
  fileName: "FilmScoreTreatment.pdf",
  communities: [
    "People experiencing grief",
    "Postpartum support networks",
    "Young adults in crisis",
    "Mental health advocates",
  ],
  topics: [
    {
      topic: "Grief and Bereavement",
      quote:
        "Dave's transformation unfolds against the backdrop of profound family loss — the sudden death of his closest friend Connor.",
      recommendations: [
        "Add an afterword citing grief support organizations (e.g., The Dougy Center, GriefShare) with direct contact information.",
        "Include a brief author's note acknowledging the emotional weight of this arc to signal care for readers navigating similar loss.",
        "Consider a sensitivity read from a grief counselor before final production to ensure the portrayal is responsible and accurate.",
      ],
    },
    {
      topic: "Postpartum Mental Health",
      quote:
        "Rachel's near-fatal pre-eclampsia and subsequent mental health crisis are woven into the narrative's darkest chapter.",
      recommendations: [
        "Add a content advisory at the opening specifically flagging postpartum distress and suicidal ideation themes.",
        "Consult with Postpartum Support International during development to strengthen authenticity and build community trust.",
        "Provide a resource card at the end of the film linking to PSI's helpline and online peer support communities.",
      ],
    },
    {
      topic: "Resilience and Second Chances",
      quote:
        "The story ultimately offers a hopeful resolution, emphasizing the enduring power of love and family across time.",
      recommendations: [
        "Develop a discussion guide framing the film's arc for educational or therapeutic group settings.",
        "Reach out to resilience-focused nonprofits as co-presenters for community screenings and Q&A events.",
        "Partner with libraries or community centers in underserved areas to offer free access to the film alongside support programming.",
      ],
    },
  ],
  overall_note:
    "This screenplay reaches audiences navigating grief, mental health challenges, and family rupture. With a few structural additions — content advisories, resource pages, and community partnerships — it has real potential to function not just as entertainment but as a point of connection and support for vulnerable readers.",
  sources: [
    "Chunk 1 — community_guidelines.pdf#3",
    "Chunk 2 — impact_framework.pdf#7",
  ],
};

// ─── Shared: Stars (identical to App.tsx) ───────────────────────────────────

function Stars({ n, max = 5 }: { n: number; max?: number }) {
  return (
    <div style={{ display: "flex", gap: 2 }}>
      {Array.from({ length: max }).map((_, i) => (
        <svg key={i} width="18" height="18" viewBox="0 0 20 20">
          <polygon
            points="10,1.5 12.5,7 18.5,7.5 14,11.5 15.5,17.5 10,14 4.5,17.5 6,11.5 1.5,7.5 7.5,7"
            fill={i < n ? C.gold : "#E8E2D8"}
            stroke={i < n ? C.gold : "#D5CFC5"}
            strokeWidth="0.5"
          />
        </svg>
      ))}
    </div>
  );
}

// ─── Card 1: Chat Response ───────────────────────────────────────────────────

function ChatResponseCard() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* User message */}
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <div
          style={{
            maxWidth: 480,
            background: C.goldSoft,
            borderRadius: "16px 16px 4px 16px",
            padding: "10px 14px",
            fontFamily: sans,
            fontSize: 14,
            color: C.ink,
            lineHeight: 1.5,
            border: `1px solid ${C.border}`,
          }}
        >
          {MOCK_CHAT.question}
        </div>
      </div>

      {/* Assistant message */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
        {/* Avatar */}
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: "50%",
            background: C.ink,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            fontFamily: sans,
            fontSize: 11,
            fontWeight: 700,
            color: C.cream,
            letterSpacing: "0.04em",
          }}
        >
          IS
        </div>

        {/* Bubble */}
        <div
          style={{
            maxWidth: 520,
            background: C.card,
            borderRadius: "4px 16px 16px 16px",
            padding: "12px 16px",
            fontFamily: font,
            fontSize: 14,
            color: C.ink,
            lineHeight: 1.65,
            border: `1px solid ${C.border}`,
            boxShadow: `0 1px 4px ${C.shadow}`,
          }}
        >
          {MOCK_CHAT.answer}

          {/* Route indicator */}
          <div
            style={{
              marginTop: 10,
              paddingTop: 8,
              borderTop: `1px solid ${C.borderLight}`,
              fontFamily: sans,
              fontSize: 10,
              color: C.inkMuted,
              display: "flex",
              alignItems: "center",
              gap: 5,
            }}
          >
            <span
              style={{
                display: "inline-block",
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: C.inkMuted,
              }}
            />
            Conversation · referenced 1 prior review
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Card 2: Script Review (original ReviewResultCard from App.tsx) ──────────

function ScriptReviewCard() {
  const r = MOCK_REVIEW;
  const yes = r.verdict === "Yes";
  return (
    <div
      style={{
        background: C.card,
        borderRadius: 16,
        border: `1px solid ${C.border}`,
        overflow: "hidden",
        boxShadow: `0 2px 12px ${C.shadow}`,
      }}
    >
      <div
        style={{
          padding: "24px 28px",
          borderBottom: `1px solid #F0D8D3`,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          background: yes
            ? "linear-gradient(135deg, #FDF8F5 0%, #FFF9F7 100%)"
            : "linear-gradient(135deg, #FDF0ED 0%, #FFF5F3 100%)",
        }}
      >
        <div>
          <div
            style={{
              fontSize: 11,
              textTransform: "uppercase" as const,
              letterSpacing: 1.5,
              color: C.accent,
              fontFamily: sans,
              marginBottom: 10,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            {/* Film reel icon */}
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
              <circle cx="8" cy="8" r="7" stroke={C.accent} strokeWidth="1.4" />
              <circle cx="8" cy="8" r="2.5" fill={C.accent} />
              <circle cx="8" cy="3" r="1.3" fill={C.accent} opacity="0.6" />
              <circle cx="8" cy="13" r="1.3" fill={C.accent} opacity="0.6" />
              <circle cx="3" cy="8" r="1.3" fill={C.accent} opacity="0.6" />
              <circle cx="13" cy="8" r="1.3" fill={C.accent} opacity="0.6" />
            </svg>
            Script Review
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <span
              style={{
                fontSize: 26,
                fontWeight: 700,
                fontFamily: font,
                color: yes ? "#2E7D6B" : C.accent,
              }}
            >
              {yes ? "Uplifting" : "Needs Work"}
            </span>
          </div>
          <div
            style={{
              marginTop: 8,
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <Stars n={r.score} />
            <span style={{ fontSize: 14, color: C.inkSoft, fontFamily: sans }}>
              {r.score} of 5
            </span>
          </div>
        </div>
        <div
          style={{
            width: 64,
            height: 64,
            borderRadius: 16,
            background: yes ? "#EEF8F5" : "#FDECEA",
            border: `1.5px solid ${yes ? "#B8DDD7" : "#F5C6C0"}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 32,
            fontWeight: 800,
            color: yes ? "#2E7D6B" : C.accent,
            fontFamily: font,
          }}
        >
          {r.score}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr" }}>
        <div
          style={{
            padding: "20px 28px",
            borderRight: `1px solid ${C.borderLight}`,
            borderBottom: `1px solid ${C.borderLight}`,
          }}
        >
          <div
            style={{
              fontSize: 11,
              textTransform: "uppercase" as const,
              letterSpacing: 1.2,
              color: C.teal,
              fontWeight: 700,
              fontFamily: sans,
              marginBottom: 14,
            }}
          >
            Strengths
          </div>
          {r.benefits.map((b, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                gap: 10,
                marginBottom: 12,
                alignItems: "flex-start",
              }}
            >
              <span
                style={{
                  color: C.teal,
                  fontSize: 14,
                  marginTop: 1,
                  flexShrink: 0,
                }}
              >
                &#9670;
              </span>
              <span
                style={{
                  fontSize: 14,
                  color: C.ink,
                  lineHeight: 1.55,
                  fontFamily: sans,
                }}
              >
                {b}
              </span>
            </div>
          ))}
        </div>
        <div
          style={{
            padding: "20px 28px",
            borderBottom: `1px solid ${C.borderLight}`,
          }}
        >
          <div
            style={{
              fontSize: 11,
              textTransform: "uppercase" as const,
              letterSpacing: 1.2,
              color: C.gold,
              fontWeight: 700,
              fontFamily: sans,
              marginBottom: 14,
            }}
          >
            Considerations
          </div>
          {r.risks.map((b, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                gap: 10,
                marginBottom: 12,
                alignItems: "flex-start",
              }}
            >
              <span
                style={{
                  color: C.gold,
                  fontSize: 14,
                  marginTop: 1,
                  flexShrink: 0,
                }}
              >
                &#9671;
              </span>
              <span
                style={{
                  fontSize: 14,
                  color: C.ink,
                  lineHeight: 1.55,
                  fontFamily: sans,
                }}
              >
                {b}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ padding: "20px 28px" }}>
        <div
          style={{
            fontSize: 11,
            textTransform: "uppercase" as const,
            letterSpacing: 1.2,
            color: C.inkMuted,
            fontWeight: 700,
            fontFamily: sans,
            marginBottom: 10,
          }}
        >
          Overall Assessment
        </div>
        <p
          style={{
            fontSize: 15,
            color: C.ink,
            lineHeight: 1.7,
            fontFamily: font,
            margin: 0,
            fontStyle: "italic",
          }}
        >
          &ldquo;{r.rationale}&rdquo;
        </p>
      </div>
    </div>
  );
}

// ─── Card 3: Impact Analysis ─────────────────────────────────────────────────
// Visual language: editorial advisory memo.
// Structure reflects the agent's actual chain of reasoning:
//   WHO is affected → EVIDENCE from text → RECOMMENDED ACTION

function ImpactTopicBlock({
  topic,
  quote,
  recommendations,
  index,
}: {
  topic: string;
  quote: string;
  recommendations: string[];
  index: number;
}) {
  return (
    <div
      style={{
        marginBottom: 20,
        paddingBottom: 20,
        borderBottom: `1px solid #D8EDEA`,
      }}
    >
      {/* Topic label */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          marginBottom: 12,
        }}
      >
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            width: 22,
            height: 22,
            borderRadius: "50%",
            background: C.teal,
            color: "#fff",
            fontSize: 11,
            fontWeight: 700,
            fontFamily: sans,
            flexShrink: 0,
          }}
        >
          {index + 1}
        </span>
        <span
          style={{
            fontFamily: sans,
            fontSize: 13,
            fontWeight: 700,
            color: C.tealMid,
            letterSpacing: "0.02em",
          }}
        >
          {topic}
        </span>
      </div>

      <div style={{ paddingLeft: 32 }}>

        {/* EVIDENCE row */}
        <div style={{ marginBottom: 12 }}>
          <div
            style={{
              fontFamily: sans,
              fontSize: 9,
              fontWeight: 700,
              letterSpacing: "0.14em",
              textTransform: "uppercase" as const,
              color: C.teal,
              marginBottom: 5,
              opacity: 0.8,
            }}
          >
            Evidence from submission
          </div>
          <blockquote
            style={{
              margin: 0,
              padding: "9px 13px",
              borderLeft: `3px solid ${C.teal}`,
              background: "rgba(42,143,130,0.06)",
              borderRadius: "0 8px 8px 0",
              fontFamily: font,
              fontSize: 13,
              fontStyle: "italic",
              color: C.inkSoft,
              lineHeight: 1.65,
            }}
          >
            &ldquo;{quote}&rdquo;
          </blockquote>
        </div>

        {/* RECOMMENDED ACTIONS list */}
        <div>
          <div
            style={{
              fontFamily: sans,
              fontSize: 9,
              fontWeight: 700,
              letterSpacing: "0.14em",
              textTransform: "uppercase" as const,
              color: C.tealMid,
              marginBottom: 8,
            }}
          >
            Recommended Actions
          </div>
          <div
            style={{
              background: C.tealSoft,
              border: `1px solid #C5DDD9`,
              borderRadius: 8,
              padding: "4px 0",
            }}
          >
            {recommendations.map((action, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  gap: 10,
                  padding: "7px 13px",
                  borderBottom: i < recommendations.length - 1
                    ? `1px solid #D8EDEA`
                    : "none",
                }}
              >
                <span
                  style={{
                    color: C.teal,
                    fontWeight: 700,
                    fontSize: 13,
                    flexShrink: 0,
                    marginTop: 1,
                  }}
                >
                  →
                </span>
                <span
                  style={{
                    fontFamily: sans,
                    fontSize: 13,
                    color: C.ink,
                    lineHeight: 1.6,
                  }}
                >
                  {action}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ImpactAnalysisCard() {
  const { fileName, communities, topics, overall_note, sources } = MOCK_IMPACT;

  return (
    <div
      style={{
        background: C.card,
        borderRadius: 16,
        border: `1px solid #C5DDD9`,
        boxShadow: `0 2px 12px ${C.shadow}`,
        overflow: "hidden",
      }}
    >
      {/* ── Header ── */}
      <div
        style={{
          padding: "24px 28px",
          borderBottom: `1px solid #C5DDD9`,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          background: "linear-gradient(135deg, #E4F4F1 0%, #F2FAF8 100%)",
        }}
      >
        <div>
          <div
            style={{
              fontSize: 11,
              textTransform: "uppercase" as const,
              letterSpacing: 1.5,
              color: C.tealMid,
              fontFamily: sans,
              marginBottom: 10,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            {/* Ripple icon — represents community impact radiating outward */}
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
              <circle cx="8" cy="8" r="2.5" fill={C.teal} />
              <circle cx="8" cy="8" r="5.5" stroke={C.teal} strokeWidth="1.2" opacity="0.5" />
              <circle cx="8" cy="8" r="7.5" stroke={C.teal} strokeWidth="0.8" opacity="0.25" />
            </svg>
            Impact Analysis
          </div>
          <div
            style={{
              fontSize: 26,
              fontWeight: 700,
              fontFamily: font,
              color: C.tealMid,
              marginBottom: 4,
            }}
          >
            Community Outreach Advisory
          </div>
          <div
            style={{
              fontFamily: sans,
              fontSize: 13,
              fontStyle: "italic",
              color: C.inkSoft,
            }}
          >
            {fileName}
          </div>
        </div>

        {/* Topic count — not a score, just a count of advisory items */}
        <div style={{ textAlign: "center" as const, flexShrink: 0 }}>
          <div
            style={{
              width: 64,
              height: 64,
              borderRadius: 16,
              background: C.tealSoft,
              border: `1.5px solid #B2D8D4`,
              display: "flex",
              flexDirection: "column" as const,
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <span
              style={{
                fontSize: 26,
                fontWeight: 800,
                color: C.teal,
                fontFamily: font,
                lineHeight: 1,
              }}
            >
              {topics.length}
            </span>
            <span
              style={{
                fontFamily: sans,
                fontSize: 9,
                color: C.tealMid,
                letterSpacing: "0.06em",
                marginTop: 2,
              }}
            >
              TOPICS
            </span>
          </div>
        </div>
      </div>

      {/* ── Communities Affected ── */}
      <div
        style={{
          padding: "16px 28px",
          borderBottom: `1px solid ${C.borderLight}`,
          background: C.bgWarm,
        }}
      >
        <div
          style={{
            fontSize: 11,
            textTransform: "uppercase" as const,
            letterSpacing: 1.2,
            color: C.inkMuted,
            fontWeight: 700,
            fontFamily: sans,
            marginBottom: 10,
          }}
        >
          Communities Affected
        </div>
        <div style={{ display: "flex", flexWrap: "wrap" as const, gap: 7 }}>
          {communities.map((c, i) => (
            <span
              key={i}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 5,
                background: C.tealSoft,
                border: `1px solid #B2D8D4`,
                borderRadius: 20,
                padding: "4px 11px",
                fontFamily: sans,
                fontSize: 12,
                color: C.tealMid,
                fontWeight: 600,
              }}
            >
              {/* Small person icon */}
              <svg width="9" height="9" viewBox="0 0 10 12" fill="none">
                <circle cx="5" cy="3" r="2.5" fill={C.teal} />
                <path d="M1 11c0-2.2 1.8-4 4-4s4 1.8 4 4" stroke={C.teal} strokeWidth="1.3" strokeLinecap="round" />
              </svg>
              {c}
            </span>
          ))}
        </div>
      </div>

      {/* ── Topics & Recommendations ── */}
      <div style={{ padding: "22px 28px", borderBottom: `1px solid ${C.borderLight}` }}>
        <div
          style={{
            fontSize: 11,
            textTransform: "uppercase" as const,
            letterSpacing: 1.2,
            color: C.teal,
            fontWeight: 700,
            fontFamily: sans,
            marginBottom: 18,
          }}
        >
          Topics &amp; Outreach Recommendations
        </div>
        {topics.map((t, i) => (
          <ImpactTopicBlock key={i} index={i} {...t} />
        ))}
      </div>

      {/* ── Overall Editorial Note ── */}
      <div style={{ padding: "20px 28px" }}>
        <div
          style={{
            fontSize: 11,
            textTransform: "uppercase" as const,
            letterSpacing: 1.2,
            color: C.inkMuted,
            fontWeight: 700,
            fontFamily: sans,
            marginBottom: 10,
          }}
        >
          Editorial Note
        </div>
        <p
          style={{
            fontSize: 15,
            color: C.ink,
            lineHeight: 1.7,
            fontFamily: font,
            margin: "0 0 16px",
            fontStyle: "italic",
          }}
        >
          &ldquo;{overall_note}&rdquo;
        </p>

        {/* ── Advisory References (RAG sources) ── */}
        {sources && sources.length > 0 && (
          <div
            style={{
              padding: "10px 14px",
              background: C.bgWarm,
              borderRadius: 10,
              border: `1px solid ${C.border}`,
            }}
          >
            <div
              style={{
                fontFamily: sans,
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: "0.12em",
                textTransform: "uppercase" as const,
                color: C.inkMuted,
                marginBottom: 6,
              }}
            >
              Advisory References
            </div>
            {sources.map((s, i) => (
              <div
                key={i}
                style={{
                  fontFamily: sans,
                  fontSize: 12,
                  color: C.inkSoft,
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  marginBottom: 3,
                }}
              >
                <span style={{ color: C.teal, fontSize: 10 }}>◆</span> {s}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Demo Page ───────────────────────────────────────────────────────────────

function CardSectionHeader({
  label,
  color,
  description,
}: {
  label: string;
  color: string;
  description: string;
}) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 7,
          background: color + "18",
          border: `1px solid ${color}44`,
          borderRadius: 20,
          padding: "4px 12px",
          marginBottom: 8,
        }}
      >
        <span
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: color,
            display: "inline-block",
          }}
        />
        <span
          style={{
            fontFamily: sans,
            fontSize: 11,
            fontWeight: 700,
            color: color,
            letterSpacing: "0.06em",
          }}
        >
          {label}
        </span>
      </div>
      <p
        style={{
          fontFamily: sans,
          fontSize: 12,
          color: C.inkMuted,
          margin: "0 0 14px",
          lineHeight: 1.6,
        }}
      >
        {description}
      </p>
    </div>
  );
}

export default function CardsDemo() {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: C.bg,
        padding: "40px 24px 80px",
        fontFamily: sans,
      }}
    >
      {/* Page title */}
      <div style={{ maxWidth: 640, margin: "0 auto 52px" }}>
        <div
          style={{
            fontFamily: sans,
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.16em",
            textTransform: "uppercase" as const,
            color: C.inkMuted,
            marginBottom: 6,
          }}
        >
          ImpactStudio · Card Design System
        </div>
        <h1
          style={{
            fontFamily: font,
            fontSize: 28,
            fontWeight: 400,
            color: C.ink,
            margin: "0 0 10px",
          }}
        >
          Response Card Designs
        </h1>
        <p
          style={{
            color: C.inkSoft,
            fontSize: 14,
            margin: 0,
            lineHeight: 1.6,
          }}
        >
          Three response types rendered by the orchestrator. Each maps to a
          different routing outcome: <strong>Chat</strong> (follow-up conversation),{" "}
          <strong>Script Review</strong> (structured critique), and{" "}
          <strong>Impact Analysis</strong> (community outreach advisory).
        </p>
      </div>

      <div
        style={{
          maxWidth: 640,
          margin: "0 auto",
          display: "flex",
          flexDirection: "column",
          gap: 56,
        }}
      >
        {/* Section 1: Chat */}
        <section>
          <CardSectionHeader
            label="Route → Chat"
            color={C.inkMuted}
            description="Triggered when the user asks a follow-up question or references a prior result without submitting new material. Gemini generates a natural language reply using chat history. No structured card — renders as a message bubble."
          />
          <ChatResponseCard />
        </section>

        {/* Section 2: Script Review */}
        <section>
          <CardSectionHeader
            label="Route → Script Review"
            color={C.accent}
            description="Triggered when the user submits a screenplay or creative document for critique. Returns a structured verdict (Uplifting / Needs Work), a 1–5 star score, strengths, considerations, and an overall assessment."
          />
          <ScriptReviewCard />
        </section>

        {/* Section 3: Impact Analysis */}
        <section>
          <CardSectionHeader
            label="Route → Impact Analysis"
            color={C.teal}
            description="Triggered when the user submits content for community impact advisory. The Impact Agent identifies which communities are affected by the content's themes, then provides specific, actionable outreach and support recommendations — with quotes from the submission as evidence. RAG-retrieved reference sources are cited when available."
          />
          <ImpactAnalysisCard />
        </section>
      </div>
    </div>
  );
}
