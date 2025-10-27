import re

def get_arxiv_link(paper_id):
    """
    Generates a valid arXiv URL from various dataset ID formats, removing any version numbers.

    This function is corrected to handle complex old-style identifiers
    (e.g., 'hep-th-9901001') and new-style identifiers correctly.
    It strips version suffixes like 'v1', 'v2', etc., from the final URL.

    Args:
        paper_id (str): The arXiv paper identifier from the dataset.

    Returns:
        str: A full, clickable URL to the latest version of the arXiv paper's abstract page,
             or an error message if it cannot be parsed.
    """
    base_url = "https://arxiv.org/abs/"
    
    # Step 1: Clean the ID by removing any potential 'abs-' prefix.
    if paper_id.lower().startswith('abs-'):
        cleaned_id = paper_id[4:]
    else:
        cleaned_id = paper_id
        
    # Step 2: Remove the version number (e.g., 'v1', 'v2') from the end.
    # We split the string by the version pattern and take the first part.
    cleaned_id = re.split(r'v\d+$', cleaned_id)[0]
        
    # Step 3: Distinguish between old and new formats.
    # New format IDs start with a number (e.g., '1501.00001').
    # Old format IDs start with a letter (e.g., 'cs-9308101').
    if cleaned_id and cleaned_id[0].isdigit():
        # This is a new format ID and can be used directly.
        return f"{base_url}{cleaned_id}"
    else:
        # This is an old format ID. The format is category/number.
        # In datasets, this is often represented as category-number.
        # The category itself can contain hyphens (e.g., hep-th).
        # We need to replace the last hyphen before the numeric part with a '/'.
        
        # Regex to find the category part and the number part.
        # The number part is 7 digits (YYMMNNN).
        match = re.match(r'(.*)-(\d{7})$', cleaned_id)
        
        if match:
            category = match.group(1)
            number_part = match.group(2)
            # The URL requires a '/' separator.
            return f"{base_url}{category}/{number_part}"
        else:
            # If the regex doesn't match, it might be an unusual format.
            # A simple replacement might be incorrect, so we return an error.
            return f"Could not parse old-style ID: {paper_id}"

# --- Example Usage ---
if __name__ == '__main__':
    # Test case 1: New paper ID with 'abs-' prefix and version
    new_paper_id = "abs-2501.18084v1"
    new_link = get_arxiv_link(new_paper_id)
    print(f"Original ID: {new_paper_id}")
    print(f"Generated Link: {new_link}\n")

    # Test case 2: Simple Old paper ID with version
    old_paper_id = "cs-9308101v1"
    old_link = get_arxiv_link(old_paper_id)
    print(f"Original ID: {old_paper_id}")
    print(f"Generated Link: {old_link}\n")
    
    # Test case 3: New paper ID without prefix or version
    new_paper_id_no_prefix = "1501.00001"
    new_link_no_prefix = get_arxiv_link(new_paper_id_no_prefix)
    print(f"Original ID: {new_paper_id_no_prefix}")
    print(f"Generated Link: {new_link_no_prefix}\n")

    # Test case 4: Complex Old paper ID with version
    complex_old_id = "hep-th-9901001v2"
    complex_link = get_arxiv_link(complex_old_id)
    print(f"Original ID: {complex_old_id}")
    print(f"Generated Link: {complex_link}\n")

# References:
# https://info.arxiv.org/help/find/index.html
# https://arxiv.org/abs/cs/9308101
# https://arxiv.org/abs/2501.18084

