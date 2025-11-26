"""
Transform CSV files into RAG-optimized markdown format.
Processes customers, projects, and quotes with semantic relationships.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path
from html.parser import HTMLParser


class HTMLStripper(HTMLParser):
    """Remove HTML tags from text."""
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, d):
        self.text.append(d)

    def get_data(self):
        return ''.join(self.text).strip()


def strip_html(text):
    """Remove HTML tags and clean up text."""
    if not text:
        return ""
    stripper = HTMLStripper()
    try:
        stripper.feed(text)
        cleaned = stripper.get_data()
        # Clean up extra whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    except:
        return text


def sanitize_filename(name):
    """Remove invalid filename characters."""
    # Remove control characters, tabs, newlines, etc.
    name = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', name)
    # Remove invalid filename characters
    name = re.sub(r'[\\/:*?"<>|]', '', name)
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Remove trailing dots and spaces
    name = name.rstrip('. ')
    # Truncate
    return name[:40]


def load_csv(file_path):
    """Load CSV file into list of dicts."""
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def transform_customers_locations(csv_path, output_path):
    """Transform customers_locations.csv into facility directory markdown."""
    locations = load_csv(csv_path)

    # Group by company
    by_company = defaultdict(list)
    for loc in locations:
        company = loc.get('name', 'Unknown')
        by_company[company].append(loc)

    output = []
    output.append("# Customer Locations Directory\n")
    output.append("Comprehensive facility and location information for all customer operations.\n")

    # Sort companies alphabetically
    for company in sorted(by_company.keys()):
        output.append(f"## {company}\n")

        locations_list = by_company[company]

        for loc in locations_list:
            location_name = loc.get('location', 'Unnamed Location')
            address = loc.get('address_full', '')
            city = loc.get('city', '')
            state = loc.get('state', '')
            postal = loc.get('postal_code', '')
            country = loc.get('country', '')

            output.append(f"### {location_name}\n")

            if address:
                output.append(f"**Address:** {address}\n")
            else:
                # Build address from components
                addr_parts = [s for s in [loc.get('address_line1'),
                                          loc.get('address_line2'),
                                          city, state, postal, country] if s]
                if addr_parts:
                    output.append(f"**Address:** {', '.join(addr_parts)}\n")

            output.append(f"**City:** {city} | **State/Province:** {state} | **Postal Code:** {postal} | **Country:** {country}\n")
            output.append("")

    output.append("---\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

    print(f"✓ Generated: {output_path}")
    return len(locations)


def transform_projects(csv_path, projects_output, output_dir):
    """Transform projects.csv into individual project markdown files."""
    projects = load_csv(csv_path)

    # Create index
    index_output = []
    index_output.append("# Projects Index\n")
    index_output.append("Overview of all projects and initiatives.\n\n")

    # Group by department
    by_dept = defaultdict(list)
    for proj in projects:
        dept = proj.get('department_name', 'Other')
        by_dept[dept].append(proj)

    for dept in sorted(by_dept.keys()):
        index_output.append(f"## {dept}\n")
        dept_projects = sorted(by_dept[dept], key=lambda p: p.get('project_name', ''))

        for proj in dept_projects:
            proj_id = proj.get('project_id', '')
            proj_name = proj.get('project_name', 'Unnamed')
            customer = proj.get('customer_name', 'Unknown')
            index_output.append(f"- **{proj_name}** (ID: {proj_id}) - {customer}\n")

        index_output.append("")

    index_output.append("---\n")

    # Write index
    with open(projects_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(index_output))

    print(f"✓ Generated: {projects_output}")

    # Generate individual project files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for proj in projects:
        proj_id = proj.get('project_id', 'unknown')
        proj_name = proj.get('project_name', 'Unnamed Project')

        # Create filename with sanitization
        safe_name = sanitize_filename(proj_name)
        filename = f"project_{proj_id}_{safe_name}.md"
        filepath = output_dir / filename

        proj_output = []
        proj_output.append(f"# {proj_name}\n")
        proj_output.append(f"**Project ID:** {proj_id}\n")

        customer = proj.get('customer_name', '')
        location = proj.get('location_name', '')
        if customer or location:
            proj_output.append(f"**Customer:** {customer} | **Location:** {location}\n")

        department = proj.get('department_name', '')
        product = proj.get('product_name', '')
        if department or product:
            proj_output.append(f"**Department:** {department} | **Product:** {product}\n")

        target_date = proj.get('target_date', '')
        completed_date = proj.get('completed_date_time', '')
        if target_date or completed_date:
            proj_output.append(f"**Target Date:** {target_date} | **Completed:** {completed_date}\n")

        pm = proj.get('project_manager', '')
        priority = proj.get('project_priority', '')
        if pm or priority:
            proj_output.append(f"**Project Manager:** {pm} | **Priority:** {priority}\n")

        proj_output.append("")

        # Description
        description = proj.get('project_description', '')
        if description:
            cleaned_desc = strip_html(description)
            proj_output.append("## Overview\n")
            proj_output.append(f"{cleaned_desc}\n\n")

        # Requirements
        requirements = []
        if proj.get('needs_phase_gate') == 'Yes':
            requirements.append("Phase Gate Review")
        if proj.get('needs_open_issues') == 'Yes':
            requirements.append("Open Issues Resolution")
        if proj.get('needs_tech_review') == 'Yes':
            requirements.append("Technical Review")
        if proj.get('needs_internal_kickoff') == 'Yes':
            requirements.append("Internal Kickoff")
        if proj.get('needs_external_kickoff') == 'Yes':
            requirements.append("External Kickoff")

        if requirements:
            proj_output.append("## Requirements\n")
            for req in requirements:
                proj_output.append(f"- {req}\n")
            proj_output.append("")

        # Key dates
        lead_time = proj.get('lead_time_due_date', '')
        if lead_time:
            proj_output.append(f"**Lead Time Due Date:** {lead_time}\n")

        proj_output.append("")
        proj_output.append("---\n")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(proj_output))

    return len(projects)


def transform_quotes(csv_path, quotes_output, output_dir):
    """Transform quotes.csv into individual quote markdown files."""
    quotes = load_csv(csv_path)

    # Create index
    index_output = []
    index_output.append("# Quotes Index\n")
    index_output.append("Overview of all customer quotes and proposals.\n\n")

    # Group by status
    by_status = defaultdict(list)
    for quote in quotes:
        status = quote.get('closed_reason', 'Open')
        if not status:
            status = "Open"
        by_status[status].append(quote)

    status_order = ['Open', 'DENIED', 'APPROVED', 'COMPLETED']
    for status in status_order:
        if status in by_status:
            index_output.append(f"## {status}\n")
            status_quotes = sorted(by_status[status], key=lambda q: q.get('quote_name', ''))

            for quote in status_quotes:
                quote_id = quote.get('quote_ID', '')
                quote_name = quote.get('quote_name', 'Unnamed')
                customer = quote.get('customer_name', 'Unknown')
                index_output.append(f"- **{quote_name}** (ID: {quote_id}) - {customer}\n")

            index_output.append("")

    index_output.append("---\n")

    # Write index
    with open(quotes_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(index_output))

    print(f"✓ Generated: {quotes_output}")

    # Generate individual quote files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for quote in quotes:
        quote_id = quote.get('quote_ID', 'unknown')
        quote_name = quote.get('quote_name', 'Unnamed Quote')

        # Create filename with sanitization
        safe_name = sanitize_filename(quote_name)
        filename = f"quote_{quote_id}_{safe_name}.md"
        filepath = output_dir / filename

        quote_output = []
        quote_output.append(f"# {quote_name}\n")
        quote_output.append(f"**Quote ID:** {quote_id}\n")

        quote_num = quote.get('quote_number', '')
        if quote_num:
            quote_output.append(f"**Quote Number:** {quote_num}\n")

        customer = quote.get('customer_name', '')
        location = quote.get('location_name', '')
        contact = quote.get('contact_name', '')
        if customer or location or contact:
            quote_output.append(f"**Customer:** {customer} | **Location:** {location}\n")
            if contact:
                quote_output.append(f"**Contact:** {contact}\n")

        department = quote.get('department_name', '')
        product = quote.get('product_name', '')
        if department or product:
            quote_output.append(f"**Department:** {department} | **Product:** {product}\n")

        quote_output.append("")

        # Dates
        added_date = quote.get('date_time_added', '')
        submitted_date = quote.get('internal_submitted_date_time', '')
        expected_due = quote.get('customer_expected_due_date', '')

        date_info = []
        if added_date:
            date_info.append(f"**Added:** {added_date}")
        if submitted_date:
            date_info.append(f"**Submitted:** {submitted_date}")
        if expected_due:
            date_info.append(f"**Expected Due:** {expected_due}")

        if date_info:
            quote_output.append(" | ".join(date_info) + "\n")

        # Status
        status = quote.get('closed_reason', 'Open')
        if not status:
            status = "Open"
        quote_output.append(f"**Status:** {status}\n")

        quote_output.append("")

        # Overview
        overview = quote.get('quote_overview', '')
        if overview:
            cleaned_overview = strip_html(overview)
            quote_output.append("## Proposal Overview\n")
            quote_output.append(f"{cleaned_overview}\n\n")

        # Technical details
        technical = quote.get('technical_detail', '')
        if technical:
            cleaned_technical = strip_html(technical)
            quote_output.append("## Technical Details\n")
            quote_output.append(f"{cleaned_technical}\n\n")

        # Background
        background = quote.get('background_information', '')
        if background:
            cleaned_bg = strip_html(background)
            quote_output.append("## Background Information\n")
            quote_output.append(f"{cleaned_bg}\n\n")

        # Purchase order information
        po_number = quote.get('purchase_order_number', '')
        po_date = quote.get('purchase_order_date', '')
        po_lead_time = quote.get('purchase_order_lead_time_due_date', '')

        if po_number or po_date or po_lead_time:
            quote_output.append("## Purchase Order Information\n")
            if po_number:
                quote_output.append(f"**PO Number:** {po_number}\n")
            if po_date:
                quote_output.append(f"**PO Date:** {po_date}\n")
            if po_lead_time:
                quote_output.append(f"**PO Lead Time Due Date:** {po_lead_time}\n")
            quote_output.append("")

        # Financial
        value = quote.get('total_proposal_value', '')
        lead_time = quote.get('lead_time_weeks', '')

        if value or lead_time:
            quote_output.append("## Commercial Information\n")
            if value:
                quote_output.append(f"**Total Proposal Value:** ${value}\n")
            if lead_time:
                quote_output.append(f"**Lead Time:** {lead_time} weeks\n")
            quote_output.append("")

        # Development info
        dev_at_nysus = quote.get('software_dev_at_nysus', '')
        dev_at_customer = quote.get('software_dev_at_customer', '')

        if dev_at_nysus or dev_at_customer:
            quote_output.append("## Development Location\n")
            if dev_at_nysus == 'Yes':
                quote_output.append("- Development at Nysus Solutions\n")
            if dev_at_customer == 'Yes':
                quote_output.append("- Development at Customer Site\n")
            quote_output.append("")

        # Submitted by
        submitted_by = quote.get('added_by_team_member_name', '')
        if submitted_by:
            quote_output.append(f"**Submitted by:** {submitted_by}\n")

        quote_output.append("")
        quote_output.append("---\n")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(quote_output))

    return len(quotes)


def main():
    """Transform all CSV files."""
    base_path = Path("data/documents/knowledge_base")

    print("\n" + "="*70)
    print("CSV to RAG-Optimized Markdown Transformer")
    print("="*70 + "\n")

    # Transform customers_locations
    print("Processing: customers_locations.csv")
    customers_csv = base_path / "customers_locations.csv"
    customers_output = base_path / "facilities_directory_rag.md"
    count = transform_customers_locations(customers_csv, customers_output)
    print(f"  ✓ Transformed {count} locations\n")

    # Transform projects
    print("Processing: projects.csv")
    projects_csv = base_path / "projects.csv"
    projects_index = base_path / "projects_index_rag.md"
    projects_dir = base_path / "projects_rag"
    count = transform_projects(projects_csv, projects_index, projects_dir)
    print(f"  ✓ Generated index and {count} individual project files\n")

    # Transform quotes
    print("Processing: quotes.csv")
    quotes_csv = base_path / "quotes.csv"
    quotes_index = base_path / "quotes_index_rag.md"
    quotes_dir = base_path / "quotes_rag"
    count = transform_quotes(quotes_csv, quotes_index, quotes_dir)
    print(f"  ✓ Generated index and {count} individual quote files\n")

    print("="*70)
    print("All transformations complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
