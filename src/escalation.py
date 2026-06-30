"""
Escalation recommendation engine.

Given a complaint category and urgency level, recommends:
    - Who to escalate to (designation)
    - Expected response timeline
    - Action text for the citizen
    - Whether it's critical
"""
from src.config import CATEGORIES, URGENCY_LEVELS

# Escalation matrix: category -> urgency -> recommendation
ESCALATION_MATRIX = {
    "Water Supply": {
        "High": {
            "escalate_to": "Municipal Commissioner / Water Board MD",
            "timeline": "24 hours",
            "action_text": "Critical water issue detected. Immediate dispatch of emergency team recommended.",
            "is_critical": True,
        },
        "Medium": {
            "escalate_to": "Zonal Water Supply Officer",
            "timeline": "48 hours",
            "action_text": "Water supply issue requires attention. Route to concerned zonal office for resolution.",
            "is_critical": False,
        },
        "Low": {
            "escalate_to": "Local Water Dept Office",
            "timeline": "7 days",
            "action_text": "Routine water complaint. Will be addressed in regular maintenance schedule.",
            "is_critical": False,
        },
    },
    "Road & Transport": {
        "High": {
            "escalate_to": "PWD Minister / Transport Commissioner",
            "timeline": "12 hours",
            "action_text": "Critical road or transport safety issue. emergency response team to be mobilized.",
            "is_critical": True,
        },
        "Medium": {
            "escalate_to": "PWD Engineer / Regional Transport Office",
            "timeline": "3 days",
            "action_text": "Road/transport issue noted. Concerned engineering team will inspect and take action.",
            "is_critical": False,
        },
        "Low": {
            "escalate_to": "Local PWD Maintenance Office",
            "timeline": "7 days",
            "action_text": "Routine road/transport complaint. Scheduled for regular maintenance.",
            "is_critical": False,
        },
    },
    "Electricity": {
        "High": {
            "escalate_to": "Power Corporation MD / Chief Engineer",
            "timeline": "4 hours",
            "action_text": "Critical power issue detected. Emergency electrical team dispatched immediately.",
            "is_critical": True,
        },
        "Medium": {
            "escalate_to": "Circle Electrical Engineer / Zonal Office",
            "timeline": "24 hours",
            "action_text": "Power supply issue noted. Concerned engineer will investigate and resolve.",
            "is_critical": False,
        },
        "Low": {
            "escalate_to": "Local Electricity Board Office",
            "timeline": "5 days",
            "action_text": "Routine electricity complaint. Scheduled for regular maintenance check.",
            "is_critical": False,
        },
    },
    "Healthcare": {
        "High": {
            "escalate_to": "Hospital Director / Chief Medical Officer",
            "timeline": "2 hours",
            "action_text": "Critical health emergency! Immediate medical response team activation required.",
            "is_critical": True,
        },
        "Medium": {
            "escalate_to": "Medical Superintendent / District Health Officer",
            "timeline": "24 hours",
            "action_text": "Healthcare issue requires attention. Concerned medical officer will follow up.",
            "is_critical": False,
        },
        "Low": {
            "escalate_to": "Hospital Administration / Health Dept",
            "timeline": "7 days",
            "action_text": "Routine healthcare feedback. Will be addressed through regular channels.",
            "is_critical": False,
        },
    },
    "Public Safety": {
        "High": {
            "escalate_to": "Police Commissioner / District Magistrate",
            "timeline": "1 hour",
            "action_text": "CRITICAL! Immediate law enforcement response required. Alert senior officials.",
            "is_critical": True,
        },
        "Medium": {
            "escalate_to": "SHO / DSP / Labour Commissioner",
            "timeline": "4 hours",
            "action_text": "Public safety concern noted. Concerned officer to investigate and take appropriate action.",
            "is_critical": False,
        },
        "Low": {
            "escalate_to": "Local Police Station / Concerned Dept",
            "timeline": "3 days",
            "action_text": "Routine public safety matter. Will be processed through normal channels.",
            "is_critical": False,
        },
    },
    "Education": {
        "High": {
            "escalate_to": "Education Secretary / Director of Education",
            "timeline": "2 days",
            "action_text": "Critical education issue. Senior education official intervention recommended.",
            "is_critical": True,
        },
        "Medium": {
            "escalate_to": "District Education Officer / Block Officer",
            "timeline": "5 days",
            "action_text": "Education-related issue noted. Concerned education officer to investigate.",
            "is_critical": False,
        },
        "Low": {
            "escalate_to": "School Administration / Local Education Office",
            "timeline": "10 days",
            "action_text": "Routine education matter. Will be addressed through regular administrative channels.",
            "is_critical": False,
        },
    },
    "Housing": {
        "High": {
            "escalate_to": "Housing Board Chairman / Municipal Commissioner",
            "timeline": "3 days",
            "action_text": "Critical housing issue! Immediate intervention from senior housing authority required.",
            "is_critical": True,
        },
        "Medium": {
            "escalate_to": "Zonal Housing Officer / PWD Executive Engineer",
            "timeline": "7 days",
            "action_text": "Housing issue noted. Concerned officer will inspect and coordinate resolution.",
            "is_critical": False,
        },
        "Low": {
            "escalate_to": "Local Housing Office / Municipal Corporation",
            "timeline": "14 days",
            "action_text": "Routine housing matter. Will be processed through regular channels.",
            "is_critical": False,
        },
    },
    "Other": {
        "High": {
            "escalate_to": "Department Secretary / Grievance Officer",
            "timeline": "3 days",
            "action_text": "High priority issue. Route to concerned department for immediate attention.",
            "is_critical": True,
        },
        "Medium": {
            "escalate_to": "Section Officer / Zonal Grievance Cell",
            "timeline": "7 days",
            "action_text": "Issue noted. Concerned department section to process and respond.",
            "is_critical": False,
        },
        "Low": {
            "escalate_to": "Grievance Redressal Cell / Helpdesk",
            "timeline": "15 days",
            "action_text": "Routine grievance registered. Will be addressed through standard procedure.",
            "is_critical": False,
        },
    },
}


def recommend_escalation(category: str, urgency: str) -> dict:
    """Get escalation recommendation for a category + urgency combination.

    Args:
        category: Complaint category (e.g., 'Water Supply', 'Healthcare')
        urgency: Urgency level ('High', 'Medium', 'Low')

    Returns:
        dict with keys: escalate_to, timeline, action_text, is_critical
    """
    # Normalize inputs
    cat = category.strip()
    urg = urgency.strip()

    # Validate
    if cat not in ESCALATION_MATRIX:
        cat = "Other"
    if urg not in ["High", "Medium", "Low"]:
        urg = "Medium"

    return dict(ESCALATION_MATRIX[cat][urg])


def generate_response(grievance_text: str, category: str, urgency: str,
                      escalation: dict = None) -> str:
    """Generate a human-readable response for the citizen.

    Args:
        grievance_text: Original complaint text
        category: Predicted category
        urgency: Predicted urgency level
        escalation: Escalation recommendation dict (optional)

    Returns:
        Formatted response string
    """
    if escalation is None:
        escalation = recommend_escalation(category, urgency)

    urgency_emoji = {"High": "CRITICAL", "Medium": "ATTENTION", "Low": "ROUTINE"}.get(urgency, "INFO")

    lines = [
        "=" * 60,
        f"  GRIEVANCE ANALYSIS REPORT",
        "=" * 60,
        "",
        f"  Category:       {category}",
        f"  Urgency Level:  {urgency} ({urgency_emoji})",
        f"  Escalate To:    {escalation['escalate_to']}",
        f"  Timeline:       {escalation['timeline']}",
        "",
        f"  Recommended Action:",
        f"  {escalation['action_text']}",
        "",
    ]

    if urgency == "High":
        lines.append(f"  FLAG: This is a CRITICAL complaint requiring immediate attention!")
        lines.append("")

    lines.extend([
        "-" * 60,
        "  Original Complaint (excerpt):",
        f"  {grievance_text[:200].strip()}...",
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


def get_category_department(category: str) -> str:
    """Get the responsible department for a category."""
    from src.config import CATEGORY_DEPT
    return CATEGORY_DEPT.get(category, "Concerned Government Department")


if __name__ == '__main__':
    # Test all 8x3 = 24 combinations
    for cat in list(ESCALATION_MATRIX.keys()):
        for urg in ["High", "Medium", "Low"]:
            rec = recommend_escalation(cat, urg)
            assert rec['escalate_to'], f"Missing escalate_to for {cat}/{urg}"
            assert rec['timeline'], f"Missing timeline for {cat}/{urg}"

    print("All 24 escalation combinations validated ✓")

    # Demo
    print(generate_response(
        "There is no water supply in our area for the past 5 days. Residents are suffering.",
        "Water Supply", "High"
    ))
