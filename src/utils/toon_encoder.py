"""
TOON (Token-Oriented Object Notation) encoder for LLM context.

TOON is a compact, token-efficient format for LLM input that:
- Uses YAML-style indentation for nested objects
- Uses CSV-style tabular layout for uniform arrays
- Reduces tokens by ~40% compared to JSON

See: https://github.com/toon-format/spec
"""

import re
from typing import Any, Optional


def needs_quoting(s: str) -> bool:
    """Check if a string needs quoting in TOON format."""
    if not s:
        return True
    # Quote if contains: spaces, colons, commas, brackets, quotes, newlines
    if re.search(r'[\s:,\[\]\{\}"\n\r\t]', s):
        return True
    # Quote reserved literals
    if s.lower() in ('true', 'false', 'null'):
        return True
    return False


def escape_string(s: str) -> str:
    """Escape a string for TOON format."""
    if s is None:
        return 'null'
    s = str(s)
    # Only allowed escapes: \\, \", \n, \r, \t
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    s = s.replace('\r', '\\r')
    s = s.replace('\t', '\\t')
    return s


def encode_value(value: Any) -> str:
    """Encode a primitive value to TOON."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        # Canonical number form
        if isinstance(value, float):
            if value == int(value):
                return str(int(value))
            return str(value).rstrip('0').rstrip('.')
        return str(value)
    # String
    s = str(value)
    if needs_quoting(s):
        return f'"{escape_string(s)}"'
    return s


def encode_table(items: list[dict], fields: list[str], delimiter: str = ',') -> str:
    """
    Encode a list of uniform objects as a TOON table.

    Format:
        items[N]{field1,field2}:
          value1,value2
          value3,value4
    """
    if not items:
        return f"items[0]:"

    n = len(items)
    header = f"[{n}]{{{delimiter.join(fields)}}}:"

    rows = []
    for item in items:
        row_values = []
        for field in fields:
            val = item.get(field, '')
            # For table cells, truncate long values and escape
            val_str = str(val) if val else ''
            if len(val_str) > 200:
                val_str = val_str[:197] + '...'
            # Escape delimiter in values
            val_str = val_str.replace(delimiter, ' ')
            val_str = val_str.replace('\n', ' ').replace('\r', ' ')
            row_values.append(val_str)
        rows.append(delimiter.join(row_values))

    return header + '\n  ' + '\n  '.join(rows)


def encode_source_variables(variables: list[dict]) -> str:
    """
    Encode source variables in TOON table format.

    Input: list of dicts with keys: name, description, labels, samples
    Output: TOON table format
    """
    if not variables:
        return "source_vars[0]:"

    # Build compact representation
    rows = []
    for var in variables:
        name = var.get('name', '?')
        desc = var.get('description') or ''

        # Compact labels
        labels = var.get('value_labels') or var.get('labels')
        labels_str = ''
        if labels and isinstance(labels, dict):
            label_items = list(labels.items())[:5]
            labels_str = ';'.join(f"{k}={v}" for k, v in label_items)

        rows.append({
            'name': name,
            'desc': desc[:150] if desc else '',
            'labels': labels_str
        })

    return 'source_vars' + encode_table(rows, ['name', 'desc', 'labels'], delimiter='|')


def encode_cses_targets(targets: dict[str, str]) -> str:
    """
    Encode CSES target variables in TOON table format.

    Input: dict mapping CSES code -> description
    Output: TOON table format
    """
    rows = [{'code': code, 'desc': desc} for code, desc in targets.items()]
    return 'cses_targets' + encode_table(rows, ['code', 'desc'], delimiter='|')


def encode_codebook_entries(entries: dict[str, str]) -> str:
    """
    Encode codebook variable definitions in TOON table format.

    Input: dict mapping variable name -> definition text
    Output: TOON table format
    """
    rows = [{'var': var, 'def': defn[:250]} for var, defn in entries.items()]
    return 'codebook' + encode_table(rows, ['var', 'def'], delimiter='|')


def encode_matching_context(
    source_vars: list[dict],
    cses_targets: dict[str, str],
    codebook_defs: dict[str, str] = None,
    assigned: set = None
) -> str:
    """
    Encode full matching context in TOON format.

    This produces a compact representation optimized for LLM token efficiency.
    """
    parts = []

    # CSES targets to match
    parts.append("## CSES Targets")
    target_rows = [{'code': c, 'desc': d} for c, d in cses_targets.items()]
    parts.append('targets' + encode_table(target_rows, ['code', 'desc'], '|'))

    # Source variables with codebook info merged
    parts.append("\n## Source Variables")
    var_rows = []
    for var in source_vars:
        name = var.get('name', '?')

        # Try codebook first, then data metadata
        desc = ''
        if codebook_defs and name in codebook_defs:
            desc = codebook_defs[name][:200]
        elif var.get('description'):
            desc = var['description'][:200]

        labels = var.get('value_labels') or {}
        labels_str = ''
        if isinstance(labels, dict) and labels:
            labels_str = ';'.join(f"{k}={v}" for k, v in list(labels.items())[:4])

        var_rows.append({
            'name': name,
            'desc': desc,
            'labels': labels_str
        })

    parts.append('vars' + encode_table(var_rows, ['name', 'desc', 'labels'], '|'))

    # Already assigned
    if assigned:
        parts.append(f"\n## Already Assigned\nexcluded: {','.join(sorted(assigned)[:30])}")

    return '\n'.join(parts)
