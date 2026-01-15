"""
SchemaLabs Universal Smart Analyzer v4.0
========================================
%100 Dynamic - Works with ANY dataset
- Intelligent column matching with similarity scoring
- Smart groupby selection based on unique ratio
- Filters out zero/null results
- Prioritizes meaningful columns over IDs
"""
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher

def similarity(a: str, b: str) -> float:
    """String similarity score (0-1)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def tokenize(text: str) -> list:
    """Tokenize text"""
    clean = re.sub(r'[^a-zA-Z0-9\s_]', ' ', text.lower())
    clean = clean.replace('_', ' ')
    return [t.strip() for t in clean.split() if len(t.strip()) > 1]

def get_query_words(query: str) -> list:
    """Extract meaningful words from query"""
    stopwords = {'what', 'is', 'the', 'by', 'each', 'for', 'all', 'in', 'of', 'to', 'and', 
                 'a', 'an', 'this', 'that', 'how', 'show', 'me', 'give', 'list', 'get',
                 'can', 'you', 'please', 'i', 'want', 'need', 'would', 'like', 'tell',
                 'display', 'find', 'with', 'from', 'are', 'was', 'be', 'been', 'has', 'have',
                 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                 'must', 'shall', 'per', 'every', 'any', 'some', 'no', 'not', 'only', 'just',
                 'also', 'very', 'much', 'many', 'more', 'most', 'less', 'least', 'few',
                 'other', 'another', 'such', 'same', 'different', 'own', 'than', 'then',
                 'so', 'too', 'as', 'if', 'or', 'but', 'because', 'when', 'where', 'while',
                 'although', 'though', 'unless', 'until', 'since', 'after', 'before'}
    tokens = tokenize(query)
    return [t for t in tokens if t not in stopwords]

def detect_aggregation(query: str) -> tuple:
    """Detect aggregation type from query"""
    q = query.lower()
    if any(w in q for w in ['average', 'avg', 'mean']):
        return 'AVERAGE', 'mean'
    elif any(w in q for w in ['maximum', 'max', 'highest', 'top', 'best', 'most', 'peak']):
        return 'MAX', 'max'
    elif any(w in q for w in ['minimum', 'min', 'lowest', 'least', 'worst', 'bottom']):
        return 'MIN', 'min'
    elif any(w in q for w in ['count', 'how many', 'number of', 'frequency']):
        return 'COUNT', 'count'
    return 'TOTAL', 'sum'

def score_column_match(col_name: str, query_words: list) -> float:
    """Score how well a column matches query words"""
    col_tokens = tokenize(col_name)
    if not col_tokens or not query_words:
        return 0.0
    
    total_score = 0.0
    for qw in query_words:
        best_match = 0.0
        for ct in col_tokens:
            if qw == ct:
                best_match = max(best_match, 1.0)
            elif qw in ct or ct in qw:
                best_match = max(best_match, 0.8)
            elif len(qw) > 3 and len(ct) > 3:
                sim = similarity(qw, ct)
                if sim > 0.6:
                    best_match = max(best_match, sim * 0.6)
        total_score += best_match
    
    return total_score / len(query_words)

def is_id_column(col_name: str) -> bool:
    """Check if column is an ID column"""
    col_lower = col_name.lower()
    # Ends with _id or is exactly 'id', 'index', etc.
    if col_lower.endswith('_id') or col_lower.endswith('id'):
        return True
    if col_lower in ['id', 'index', 'idx', 'key', 'code', 'uuid', 'guid']:
        return True
    if col_lower.startswith('id_') or col_lower.startswith('index_'):
        return True
    return False

def is_name_column(col_name: str) -> bool:
    """Check if column is a name/label column (good for groupby)"""
    col_lower = col_name.lower()
    indicators = ['name', 'title', 'label', 'description']
    return any(ind in col_lower for ind in indicators)

def find_metric_columns(df: pd.DataFrame, query_words: list) -> list:
    """Find best numeric columns for the query"""
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not num_cols:
        return []
    
    scored = []
    for col in num_cols:
        # Skip ID columns unless explicitly asked
        if is_id_column(col) and not any(qw in col.lower() for qw in query_words):
            continue
        
        score = score_column_match(col, query_words)
        
        # Bonus for columns starting/ending with query word
        col_lower = col.lower()
        for qw in query_words:
            if col_lower.startswith(qw) or col_lower.endswith(qw):
                score += 0.3
        
        # Small penalty for columns with mostly zeros
        try:
            zero_ratio = (df[col] == 0).sum() / len(df)
            if zero_ratio > 0.9:
                score -= 0.2
        except:
            pass
        
        if score > 0:
            scored.append((col, score))
    
    scored.sort(key=lambda x: -x[1])
    top_cols = [c for c, s in scored]
    
    # Fallback: non-ID columns with variance
    if not top_cols:
        for col in num_cols:
            if not is_id_column(col):
                try:
                    if df[col].std() > 0:
                        top_cols.append(col)
                except:
                    pass
    
    return top_cols[:5]

def find_groupby_column(df: pd.DataFrame, query_words: list) -> str:
    """Find best categorical column for groupby"""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols:
        return None
    
    scored = []
    for col in cat_cols:
        score = score_column_match(col, query_words)
        
        # Strong bonus for name columns
        if is_name_column(col):
            score += 1.0
        
        # Penalty for ID columns
        if is_id_column(col):
            score -= 0.5
        
        # Analyze unique ratio - ideal is between 0.01 and 0.5
        try:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:  # Too many unique = probably ID
                score -= 0.8
            elif unique_ratio > 0.5:
                score -= 0.3
            elif unique_ratio < 0.01:  # Too few unique = not useful
                score -= 0.2
            elif 0.005 < unique_ratio < 0.2:  # Sweet spot
                score += 0.3
        except:
            pass
        
        scored.append((col, score))
    
    scored.sort(key=lambda x: -x[1])
    
    if scored and scored[0][1] > 0:
        return scored[0][0]
    
    # Fallback: prefer name columns
    for col in cat_cols:
        if is_name_column(col):
            return col
    
    # Last resort: first non-ID categorical with reasonable unique count
    for col in cat_cols:
        if not is_id_column(col):
            try:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:
                    return col
            except:
                return col
    
    return cat_cols[0] if cat_cols else None

def smart_analyze(df: pd.DataFrame, query: str) -> str:
    """
    Main analysis function - 100% dynamic
    Works with any dataset structure
    """
    query_words = get_query_words(query)
    agg_name, agg_func = detect_aggregation(query)
    metric_cols = find_metric_columns(df, query_words)
    groupby_col = find_groupby_column(df, query_words)
    
    # Build analysis
    analysis = "=== QUERY ANALYSIS ===\n"
    analysis += f"Dataset: {len(df):,} rows, {len(df.columns)} columns\n"
    analysis += f"Query keywords: {', '.join(query_words) if query_words else 'general'}\n"
    analysis += f"Aggregation: {agg_name}\n"
    analysis += f"Metrics: {', '.join(metric_cols[:3]) if metric_cols else 'auto-selected'}\n"
    analysis += f"Group by: {groupby_col}\n\n"
    
    # Grouped analysis
    if metric_cols and groupby_col:
        for col in metric_cols[:3]:
            try:
                grouped = df.groupby(groupby_col)[col]
                
                if agg_func == 'sum':
                    result = grouped.sum()
                elif agg_func == 'mean':
                    result = grouped.mean()
                elif agg_func == 'max':
                    result = grouped.max()
                elif agg_func == 'min':
                    result = grouped.min()
                elif agg_func == 'count':
                    result = grouped.count()
                else:
                    result = grouped.sum()
                
                # Sort
                ascending = (agg_func == 'min')
                result = result.sort_values(ascending=ascending)
                
                # Filter out NaN and zero values
                result = result[result.notna() & (result != 0)]
                
                if len(result) > 0:
                    analysis += f"{agg_name} {col} BY {groupby_col}:\n"
                    for idx, val in list(result.items())[:20]:
                        if pd.notna(val) and val != 0:
                            if abs(val) >= 10000:
                                analysis += f"  {idx}: {val:,.0f}\n"
                            elif abs(val) >= 10:
                                analysis += f"  {idx}: {val:.2f}\n"
                            elif abs(val) >= 0.01:
                                analysis += f"  {idx}: {val:.4f}\n"
                            else:
                                analysis += f"  {idx}: {val:.6f}\n"
                    analysis += "\n"
            except Exception as e:
                continue
    
    # Summary statistics
    if metric_cols:
        analysis += "=== SUMMARY ===\n"
        for col in metric_cols[:3]:
            try:
                non_zero = df[col][df[col] != 0]
                if len(non_zero) > 0:
                    total = non_zero.sum()
                    avg = non_zero.mean()
                    mx = non_zero.max()
                    mn = non_zero.min()
                    if abs(total) >= 10000:
                        analysis += f"{col}: total={total:,.0f}, avg={avg:,.0f}, max={mx:,.0f}, min={mn:,.0f}\n"
                    else:
                        analysis += f"{col}: total={total:.2f}, avg={avg:.2f}, max={mx:.2f}, min={mn:.2f}\n"
            except:
                pass
    
    # Help info if no matches
    if not metric_cols:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        non_id = [c for c in num_cols if not is_id_column(c)][:20]
        analysis += "\n=== AVAILABLE NUMERIC COLUMNS ===\n"
        analysis += f"{', '.join(non_id)}\n"
    
    if not groupby_col:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        non_id = [c for c in cat_cols if not is_id_column(c)][:20]
        analysis += "\n=== AVAILABLE CATEGORICAL COLUMNS ===\n"
        analysis += f"{', '.join(non_id)}\n"
    
    return analysis

def detect_and_analyze(df: pd.DataFrame, query: str) -> str:
    """Alias for smart_analyze"""
    return smart_analyze(df, query)
