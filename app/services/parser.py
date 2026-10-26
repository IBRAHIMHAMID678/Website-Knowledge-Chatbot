import re
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup

def table_to_markdown(table_tag) -> str:
    """Converts an HTML <table> into clean Markdown format to preserve tabular prices."""
    rows = []
    headers = []
    
    # Process thead or first tr for headers
    thead = table_tag.find("thead")
    if thead:
        th_tags = thead.find_all(["th", "td"])
        headers = [re.sub(r"\s+", " ", th.get_text().strip()) for th in th_tags]
    
    tr_tags = table_tag.find_all("tr")
    start_idx = 0
    if not headers and tr_tags:
        first_row_ths = tr_tags[0].find_all(["th", "td"])
        headers = [re.sub(r"\s+", " ", th.get_text().strip()) for th in first_row_ths]
        start_idx = 1
        
    for tr in tr_tags[start_idx:]:
        cols = tr.find_all(["td", "th"])
        if cols:
            row_data = [re.sub(r"\s+", " ", col.get_text().strip()) for col in cols]
            # Skip empty rows
            if any(row_data):
                rows.append(row_data)
                
    if not headers and not rows:
        return ""
        
    # Standardize column counts
    max_cols = max([len(headers)] + [len(r) for r in rows]) if (headers or rows) else 0
    if max_cols == 0:
        return ""
        
    while len(headers) < max_cols:
        headers.append(f"Col {len(headers)+1}")
        
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    
    for r in rows:
        while len(r) < max_cols:
            r.append("")
        md_lines.append("| " + " | ".join(r) + " |")
        
    return "\n" + "\n".join(md_lines) + "\n"

def determine_page_type(url: str, title: str) -> str:
    u = url.lower()
    if "/profile/" in u:
        return "Clinic Profile"
    elif "/package/" in u:
        return "Treatment Package"
    elif "/profile-videos/" in u:
        return "Clinic Videos"
    elif "/price-list/" in u:
        return "Price List"
    elif "/search-medical-pricings/" in u:
        return "Price Comparison"
    return "Medical Information"

def parse_placidway_html(html_content: str, url: str) -> Dict[str, Any]:
    """
    Cleans raw PlacidWay HTML by stripping scripts, styles, navigation,
    footers, and cookie banners while extracting rich text and tables.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 1. Extract Page Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
    if not title:
        title = url.split("/")[-1].replace("-", " ")
        
    # Clean up standard suffix
    title = re.sub(r"\s*\|\s*PlacidWay.*$", "", title, flags=re.IGNORECASE)
    
    page_type = determine_page_type(url, title)
    
    # 2. Decompose non-content boilerplate
    for tag in soup([
        "script", "style", "noscript", "svg", "iframe",
        "header", "nav", "footer"
    ]):
        tag.decompose()
        
    # Remove cookie consent banners and known widgets
    for junk in soup.find_all(class_=re.compile(r"(cookie|consent|cky-|hotjar|user-menu|breadcrumb|modal|social-share|share-bar)", re.I)):
        junk.decompose()

    # 3. Replace HTML tables with Markdown representations wrapped in <p> tags
    for table in soup.find_all("table"):
        md_table = table_to_markdown(table)
        if md_table.strip():
            table_wrapper = soup.new_tag("p")
            table_wrapper.string = f"\n\n{md_table}\n\n"
            table.replace_with(table_wrapper)
        else:
            table.decompose()

    # 4. Extract structured sections, filtering cross-promotional boilerplate
    BOILERPLATE_HEADINGS = [
        "related experiences",
        "cancer treatment solutions near you",
        "unlock the best cancer treatment packages",
        "transformative cancer treatment experiences",
        "explore : stories, testimonials, and procedures",
        "explore: stories, testimonials, and procedures",
        "learn more with these cancer treatment articles",
        "we collaborate with internationally accredited hospitals",
        "trusted medical tourism platform since 2007"
    ]

    lines = []
    skip_section = False

    for elem in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        txt = elem.get_text(separator=" ", strip=True)
        txt = re.sub(r"\s+", " ", txt)
        if len(txt) <= 5:
            continue

        lower_txt = txt.lower()

        # Check if this heading starts a cross-promotional boilerplate section
        if elem.name.startswith("h"):
            if any(b in lower_txt for b in BOILERPLATE_HEADINGS):
                skip_section = True
                continue
            elif "about package" in lower_txt or "overview" in lower_txt or "faqs" in lower_txt:
                skip_section = False

        if skip_section:
            # Check if this element restores genuine package metadata
            if "associated center:" in lower_txt or "location: blvd." in lower_txt or "focus area:" in lower_txt:
                skip_section = False
            else:
                continue

        # Skip cross-promo inline lines
        if "related experiences:" in lower_txt or "turkey moh" in lower_txt or "korea mohw" in lower_txt or "joint commission international" in lower_txt:
            continue

        if elem.name.startswith("h"):
            lines.append(f"\n### {txt}\n")
        elif elem.name == "li":
            lines.append(f"- {txt}")
        elif "|" in txt:  # Preserve markdown table rows
            lines.append(f"\n{txt}\n")
        else:
            lines.append(txt)

    content_text = "\n".join(lines)
    # Clean multiple blank lines
    content_text = re.sub(r"\n{3,}", "\n\n", content_text).strip()
    
    return {
        "url": url,
        "title": title,
        "page_type": page_type,
        "text": content_text
    }

def chunk_document(
    doc: Dict[str, Any], 
    chunk_size: int = 750, 
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Chunks parsed document into semantic pieces with context headers.
    Ensures URL and title are attached to each chunk for strict grounding.
    """
    text = doc["text"]
    title = doc["title"]
    url = doc["url"]
    page_type = doc["page_type"]
    
    # Context header prepended to every chunk to retain grounding context
    header = f"[Source: {title} | Type: {page_type} | URL: {url}]\n"
    
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        p_len = len(para)
        if current_len + p_len > chunk_size and current_chunk:
            chunk_body = "\n\n".join(current_chunk)
            chunks.append({
                "chunk_id": f"{url}_{len(chunks)}",
                "text": header + chunk_body,
                "clean_content": chunk_body,
                "url": url,
                "title": title,
                "page_type": page_type
            })
            # Overlap: keep last paragraph if reasonable
            if len(current_chunk) > 1 and len(current_chunk[-1]) < chunk_overlap:
                current_chunk = [current_chunk[-1], para]
                current_len = len(current_chunk[0]) + p_len
            else:
                current_chunk = [para]
                current_len = p_len
        else:
            current_chunk.append(para)
            current_len += p_len
            
    if current_chunk:
        chunk_body = "\n\n".join(current_chunk)
        chunks.append({
            "chunk_id": f"{url}_{len(chunks)}",
            "text": header + chunk_body,
            "clean_content": chunk_body,
            "url": url,
            "title": title,
            "page_type": page_type
        })
        
    return chunks
