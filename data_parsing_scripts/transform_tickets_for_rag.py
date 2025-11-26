"""
Transform ticket export files into RAG-optimized format.
Groups related ticket entries together and removes redundancy.
"""

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_ticket_entry(entry_text):
    """Parse a single ticket entry block into structured data."""
    entry = {}
    lines = entry_text.strip().split('\n')

    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            entry[key.strip()] = value.strip()

    return entry


def group_tickets_by_id(file_path):
    """Read ticket file and group all entries by ticket_id."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by *** delimiter
    entries = content.split('***\n')

    tickets = defaultdict(list)

    for entry_text in entries:
        if entry_text.strip():
            entry = parse_ticket_entry(entry_text)
            if 'ticket_id' in entry:
                ticket_id = entry['ticket_id']
                tickets[ticket_id].append(entry)

    return tickets


def format_datetime(dt_str):
    """Format datetime string to be more readable."""
    if not dt_str:
        return ""
    try:
        dt = datetime.fromisoformat(dt_str.replace('+00:00', ''))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return dt_str


def clean_company_name(company_str):
    """Extract company name from JSON-like format."""
    if not company_str:
        return ""
    # Remove JSON formatting: {"Company Name":"Company Name"} -> Company Name
    match = re.search(r'"\s*([^"]+?)\s*"', company_str)
    if match:
        return match.group(1)
    return company_str


def format_attachments(attachments_str):
    """Format attachments list into readable format."""
    if not attachments_str or attachments_str == '[]':
        return None

    # Parse JSON-like attachment format
    attachments = []
    matches = re.findall(r'"filename"\s*:\s*"([^"]+)"\s*,\s*"size_bytes"\s*:\s*(\d+)', attachments_str)
    for filename, size in matches:
        attachments.append(f"{filename} ({size} bytes)")

    return attachments if attachments else None


def format_ticket_thread(ticket_id, entries):
    """Format all entries for a single ticket into RAG-optimized markdown format."""
    if not entries:
        return ""

    # Sort entries by created_at timestamp
    entries.sort(key=lambda x: x.get('created_at', ''))

    # Use first entry for ticket metadata
    first_entry = entries[0]

    ticket_number = first_entry.get('ticket_number', 'Unknown')
    company = clean_company_name(first_entry.get('company_name', ''))
    department = first_entry.get('department', '')
    help_topic = first_entry.get('help_topic', '')
    created = format_datetime(first_entry.get('ticket_created', ''))
    updated = format_datetime(first_entry.get('ticket_updated', ''))
    url = first_entry.get('ticket_url', '')
    priority = first_entry.get('priority', '')
    status = first_entry.get('status', '')

    # Build markdown header
    output = []
    output.append(f"## Ticket #{ticket_number} (ID: {ticket_id})")
    output.append("")
    output.append(f"**Company:** {company}  ")

    topic_parts = [f"**Department:** {department}"]
    if help_topic:
        topic_parts.append(f"**Topic:** {help_topic}")
    if priority:
        topic_parts.append(f"**Priority:** {priority}")
    if status:
        topic_parts.append(f"**Status:** {status}")
    output.append(" | ".join(topic_parts) + "  ")

    output.append(f"**Created:** {created} | **Updated:** {updated}  ")
    output.append(f"**URL:** {url}")
    output.append("")

    # Add each entry in chronological order
    for idx, entry in enumerate(entries):
        entry_type = entry.get('entry_type', 'message').upper()
        created_at = format_datetime(entry.get('created_at', ''))
        author_name = entry.get('author_name', 'Unknown')
        author_email = entry.get('author_email', '')
        author_type = entry.get('author_type', '')

        # Determine entry label
        if idx == 0 and entry_type == 'MESSAGE':
            entry_label = "🎫 Initial Request"
        elif entry_type == 'RESPONSE':
            entry_label = "💬 Response"
        elif entry_type == 'NOTE':
            entry_label = "📝 Internal Note"
        else:
            entry_label = entry_type

        # Format author info
        author_info = author_name
        if author_email:
            author_info += f" ({author_email})"
        elif author_type:
            author_info += f" ({author_type})"

        output.append(f"### {entry_label}")
        output.append(f"**Date:** {created_at} | **Author:** {author_info}")
        output.append("")

        # Use body_text (preferred) or body_html
        body = entry.get('body_text', entry.get('body_html', '')).strip()
        if body:
            # Clean up escaped newlines
            body = body.replace('\\n', '\n')
            output.append(body)
            output.append("")

        # Add attachments if present
        attachments = format_attachments(entry.get('attachments', ''))
        if attachments:
            output.append(f"**Attachments:** {', '.join(attachments)}")
            output.append("")

    output.append("---")
    output.append("")

    return '\n'.join(output)


def transform_file(input_path, output_path=None):
    """Transform a ticket export file to RAG-optimized markdown format."""
    input_path = Path(input_path)

    if output_path is None:
        # Create output in same directory with _rag.md suffix
        output_path = input_path.parent / f"{input_path.stem}_rag.md"
    else:
        output_path = Path(output_path)

    print(f"Processing: {input_path}")
    print(f"Output to: {output_path}")

    # Group tickets by ID
    tickets = group_tickets_by_id(input_path)
    print(f"Found {len(tickets)} unique tickets")

    # Sort ticket IDs numerically
    sorted_ticket_ids = sorted(tickets.keys(), key=lambda x: int(x))

    # Write transformed output
    with open(output_path, 'w', encoding='utf-8') as f:
        # Add markdown title
        f.write(f"# Support Tickets - {input_path.stem.replace('ost_export_', '').upper()}\n\n")

        for ticket_id in sorted_ticket_ids:
            entries = tickets[ticket_id]
            formatted = format_ticket_thread(ticket_id, entries)
            f.write(formatted)

    print(f"✓ Transformation complete!")
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transform_tickets_for_rag.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    transform_file(input_file, output_file)
