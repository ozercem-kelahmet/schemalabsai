"""
SchemaLabs Comprehensive Analytics Engine
150+ Analytics Types for Professional Data Analysis
FULLY DYNAMIC - Works with ANY dataset
"""

import pandas as pd
import numpy as np
import re

# === ANALYTICS TYPE DEFINITIONS ===
ANALYTICS_TYPES = {
    # STRATEGIC (1-20)
    'swot': ['swot', 'strength', 'weakness', 'opportunity', 'threat'],
    'pestle': ['pestle', 'political', 'economic', 'social', 'technological', 'legal', 'environmental'],
    'porter': ['porter', 'five forces', 'competitive', 'rivalry'],
    'bcg': ['bcg', 'star', 'cash cow', 'question mark'],
    'ansoff': ['ansoff', 'market penetration', 'diversification'],
    'gap': ['gap analysis', 'current vs desired', 'improvement needed'],
    'scenario': ['scenario', 'what if', 'future state', 'hypothetical'],
    'sensitivity': ['sensitivity', 'impact of change', 'variable impact'],
    'simulation': ['simulation', 'monte carlo', 'random'],
    'decision': ['decision tree', 'decision analysis', 'option'],
    'cost_benefit': ['cost benefit', 'pros cons', 'trade off'],
    'feasibility': ['feasibility', 'viable', 'practical'],
    'prioritization': ['prioritize', 'priority', 'importance ranking'],
    'roadmap': ['roadmap', 'timeline', 'milestones'],
    'maturity': ['maturity', 'capability level', 'evolution'],
    'readiness': ['readiness', 'prepared', 'capable'],
    'alignment': ['alignment', 'fit', 'consistency'],
    'synergy': ['synergy', 'combined effect', 'interaction'],
    'trade_off': ['trade off', 'compromise', 'balance'],
    'optimization': ['optimize', 'maximize', 'minimize', 'best'],
    
    # RISK (21-40)
    'risk_matrix': ['risk matrix', 'risk assessment', 'likelihood impact'],
    'risk_register': ['risk register', 'risk catalog', 'risk list'],
    'fmea': ['fmea', 'failure mode', 'effect analysis'],
    'root_cause': ['root cause', 'why analysis', 'fishbone', 'ishikawa'],
    'fault_tree': ['fault tree', 'failure path'],
    'bow_tie': ['bow tie', 'barrier analysis'],
    'risk_appetite': ['risk appetite', 'tolerance', 'acceptance'],
    'var': ['value at risk', 'var', 'potential loss'],
    'stress_test': ['stress test', 'extreme', 'worst case'],
    'heat_map': ['heat map', 'risk visualization'],
    'control': ['control assessment', 'mitigation', 'safeguard'],
    'incident': ['incident', 'accident', 'event analysis'],
    'hazard': ['hazard', 'danger', 'safety'],
    'vulnerability': ['vulnerability', 'exposure', 'weak point'],
    'threat': ['threat assessment', 'adversary', 'attack'],
    'impact': ['impact analysis', 'consequence', 'effect'],
    'probability': ['probability', 'likelihood', 'chance'],
    'exposure': ['exposure', 'at risk', 'vulnerable'],
    'residual': ['residual risk', 'remaining', 'after mitigation'],
    'inherent': ['inherent risk', 'before controls', 'gross risk'],
    
    # PERFORMANCE (41-60)
    'benchmark': ['benchmark', 'compare to average', 'vs standard', 'peer comparison'],
    'kpi': ['kpi', 'key performance', 'metric', 'indicator'],
    'scorecard': ['scorecard', 'dashboard', 'performance card'],
    'okr': ['okr', 'objective', 'key result'],
    'efficiency': ['efficiency', 'productivity', 'output input'],
    'effectiveness': ['effectiveness', 'achievement', 'success rate'],
    'utilization': ['utilization', 'usage', 'capacity'],
    'throughput': ['throughput', 'flow', 'processing'],
    'cycle_time': ['cycle time', 'lead time', 'duration'],
    'bottleneck': ['bottleneck', 'constraint', 'limiting'],
    'pareto': ['pareto', '80 20', 'vital few'],
    'variance': ['variance', 'deviation', 'difference from plan'],
    'roi': ['roi', 'return on investment', 'payback'],
    'margin': ['margin', 'profit margin', 'markup'],
    'yield': ['yield', 'output rate', 'production'],
    'quality': ['quality', 'defect', 'error rate'],
    'availability': ['availability', 'uptime', 'reliability'],
    'sla': ['sla', 'service level', 'agreement'],
    'target': ['target', 'goal', 'objective tracking'],
    'progress': ['progress', 'completion', 'status'],
    
    # STATISTICAL (61-80)
    'descriptive': ['summary', 'descriptive', 'overview', 'basic stats'],
    'correlation': ['correlation', 'relationship', 'association', 'connected'],
    'regression': ['regression', 'predict', 'model', 'forecast'],
    'anova': ['anova', 'group comparison', 'variance between'],
    'ttest': ['t test', 'mean difference', 'significant'],
    'chi_square': ['chi square', 'categorical', 'independence'],
    'distribution': ['distribution', 'histogram', 'spread'],
    'percentile': ['percentile', 'quartile', 'rank position'],
    'outlier': ['outlier', 'anomaly', 'unusual', 'exception', 'abnormal'],
    'cluster': ['cluster', 'segment', 'group', 'categorize'],
    'pca': ['pca', 'principal component', 'dimension'],
    'trend': ['trend', 'over time', 'direction', 'trajectory'],
    'seasonality': ['seasonality', 'cyclical', 'periodic'],
    'moving_avg': ['moving average', 'rolling', 'smoothing'],
    'forecast': ['forecast', 'predict', 'projection', 'future'],
    'confidence': ['confidence interval', 'margin of error'],
    'significance': ['significance', 'p value', 'statistical'],
    'sample': ['sample', 'subset', 'representative'],
    'population': ['population', 'entire', 'all data'],
    'hypothesis': ['hypothesis', 'test', 'assumption'],
    
    # COMPARATIVE (81-100)
    'ranking': ['ranking', 'rank', 'top', 'bottom', 'best', 'worst', 'highest', 'lowest'],
    'comparison': ['compare', 'versus', 'vs', 'difference between'],
    'peer': ['peer', 'similar', 'comparable', 'like'],
    'competitive': ['competitive', 'competitor', 'market position'],
    'yoy': ['year over year', 'yoy', 'annual change'],
    'mom': ['month over month', 'mom', 'monthly change'],
    'period': ['period comparison', 'before after', 'change'],
    'cohort': ['cohort', 'group over time', 'generation'],
    'league': ['league table', 'standings', 'leaderboard'],
    'quadrant': ['quadrant', 'matrix', 'four box'],
    'waterfall': ['waterfall', 'bridge', 'breakdown'],
    'attribution': ['attribution', 'contribution', 'driver'],
    'decomposition': ['decomposition', 'component', 'breakdown'],
    'composition': ['composition', 'makeup', 'structure'],
    'share': ['share', 'proportion', 'percentage of total'],
    'concentration': ['concentration', 'dominance', 'spread'],
    'diversity': ['diversity', 'variety', 'heterogeneity'],
    'balance': ['balance', 'equilibrium', 'proportion'],
    'ratio': ['ratio', 'relative', 'per unit'],
    'index': ['index', 'score', 'composite'],
    
    # SPORTS SPECIFIC (101-120)
    'player_profile': ['player profile', 'player stats', 'individual analysis'],
    'team_analysis': ['team analysis', 'squad', 'roster'],
    'match_analysis': ['match analysis', 'game review', 'fixture'],
    'fatigue': ['fatigue', 'tiredness', 'workload', 'load'],
    'injury_risk': ['injury risk', 'injury prediction', 'health'],
    'form': ['form', 'recent performance', 'momentum'],
    'fitness': ['fitness', 'conditioning', 'physical'],
    'tactical': ['tactical', 'strategy', 'formation'],
    'opponent': ['opponent', 'opposition', 'enemy'],
    'position': ['position', 'role', 'playing position'],
    'minutes': ['minutes', 'playing time', 'game time'],
    'distance': ['distance', 'distance covered', 'running'],
    'speed': ['speed', 'velocity', 'pace'],
    'sprint': ['sprint', 'high speed', 'acceleration'],
    'intensity': ['intensity', 'high intensity', 'effort'],
    'recovery': ['recovery', 'rest', 'regeneration'],
    'xg': ['xg', 'expected goals', 'shot quality'],
    'passing': ['passing', 'pass accuracy', 'distribution'],
    'shooting': ['shooting', 'shot', 'goal scoring'],
    'defensive': ['defensive', 'tackle', 'interception'],
    
    # FINANCIAL (121-140)
    'profitability': ['profitability', 'profit', 'earnings'],
    'liquidity': ['liquidity', 'cash', 'current ratio'],
    'solvency': ['solvency', 'debt', 'leverage'],
    'valuation': ['valuation', 'worth', 'value'],
    'cashflow': ['cashflow', 'cash flow', 'inflow outflow'],
    'budget': ['budget', 'spending', 'allocation'],
    'revenue': ['revenue', 'sales', 'income'],
    'cost': ['cost', 'expense', 'spending'],
    'investment': ['investment', 'capital', 'funding'],
    'return': ['return', 'gain', 'profit'],
    'growth': ['growth', 'increase', 'expansion'],
    'decline': ['decline', 'decrease', 'reduction'],
    'break_even': ['break even', 'profitability point'],
    'payback': ['payback', 'recovery period'],
    'npv': ['npv', 'net present value'],
    'irr': ['irr', 'internal rate of return'],
    'ebitda': ['ebitda', 'operating profit'],
    'gross_margin': ['gross margin', 'gross profit'],
    'net_margin': ['net margin', 'net profit'],
    'working_capital': ['working capital', 'current assets'],
    
    # CUSTOMER/MARKETING (141-160)
    'segmentation': ['segmentation', 'segment', 'customer group'],
    'funnel': ['funnel', 'conversion', 'pipeline'],
    'churn': ['churn', 'attrition', 'loss'],
    'retention': ['retention', 'keep', 'loyalty'],
    'ltv': ['ltv', 'lifetime value', 'customer value'],
    'acquisition': ['acquisition', 'new customer', 'cac'],
    'engagement': ['engagement', 'interaction', 'active'],
    'satisfaction': ['satisfaction', 'happy', 'csat'],
    'nps': ['nps', 'net promoter', 'recommend'],
    'sentiment': ['sentiment', 'opinion', 'feeling'],
    'brand': ['brand', 'awareness', 'perception'],
    'campaign': ['campaign', 'marketing', 'promotion'],
    'channel': ['channel', 'medium', 'platform'],
    'reach': ['reach', 'audience', 'impressions'],
    'frequency': ['frequency', 'how often', 'repetition'],
    'recency': ['recency', 'last activity', 'recent'],
    'rfm': ['rfm', 'recency frequency monetary'],
    'basket': ['basket', 'cross sell', 'bundle'],
    'affinity': ['affinity', 'association', 'together'],
    'recommendation': ['recommendation', 'suggest', 'next best'],
}


def detect_analytics_type(query):
    """Detect which analytics types match the query"""
    query_lower = query.lower()
    detected = []
    
    for analytics_type, keywords in ANALYTICS_TYPES.items():
        score = 0
        matched_keywords = []
        for kw in keywords:
            if kw in query_lower:
                score += len(kw)
                matched_keywords.append(kw)
        if score > 0:
            detected.append({
                'type': analytics_type,
                'score': score,
                'matched': matched_keywords
            })
    
    detected.sort(key=lambda x: -x['score'])
    return detected[:5]


def extract_entity(df, query, cat_cols):
    """
    FULLY DYNAMIC entity extraction - works with ANY dataset
    First searches for query terms in ALL columns, then returns best match
    """
    query_lower = query.lower()
    query_words = set()
    for w in re.split(r'[\s,\.\?!]+', query_lower):
        if len(w) > 2:
            query_words.add(w.strip())
    
    best_match = None
    best_score = 0
    
    # FIRST: Search ALL categorical columns for query match
    for col in cat_cols:
        try:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 500:
                unique_vals = unique_vals[:500]
            
            for val in unique_vals:
                val_str = str(val).strip()
                val_lower = val_str.lower()
                
                # Skip pure numbers and short values
                if len(val_str) < 2:
                    continue
                if val_lower.replace('.', '').replace('-', '').replace(' ', '').replace('_', '').isdigit():
                    continue
                
                score = 0
                
                # 1. Exact full match (highest priority)
                if val_lower in query_lower:
                    score = 200 + len(val_lower)
                
                # 2. Word-by-word matching
                val_words = set()
                for w in re.split(r'[\s_\-\.]+', val_lower):
                    if len(w) > 1:
                        val_words.add(w)
                
                if val_words and query_words:
                    matches = val_words & query_words
                    if matches:
                        # More matches = higher score
                        score += len(matches) * 30
                        # Bonus if most words match
                        match_ratio = len(matches) / len(val_words)
                        if match_ratio >= 0.5:
                            score += 50
                        if match_ratio == 1.0:
                            score += 100
                
                # 3. Substring matching
                for qw in query_words:
                    if len(qw) > 3 and qw in val_lower:
                        score += 20
                    # Also check if value word is in query
                    for vw in val_words:
                        if len(vw) > 3 and vw in query_lower:
                            score += 20
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        'name': val_str,
                        'column': col,
                        'data': df[df[col] == val],
                        'score': score
                    }
        except:
            continue
    
    if best_match and best_score >= 30:
        return best_match
    
    return None


def generate_swot(df, query, num_cols, cat_cols):
    """Generate SWOT analysis"""
    analysis = "=== SWOT ANALYSIS ===\n"
    entity = extract_entity(df, query, cat_cols)
    
    if entity:
        analysis += f"Entity: {entity['name']}\n\n"
        entity_data = entity['data']
        strengths, weaknesses = [], []
        
        for col in num_cols[:15]:
            try:
                entity_val = entity_data[col].mean()
                overall_avg = df[col].mean()
                overall_std = df[col].std()
                
                if entity_val > overall_avg + 0.5 * overall_std:
                    diff = ((entity_val - overall_avg) / overall_avg * 100) if overall_avg != 0 else 0
                    strengths.append(f"{col}: {entity_val:.2f} (+{diff:.1f}% above avg)")
                elif entity_val < overall_avg - 0.5 * overall_std:
                    diff = ((overall_avg - entity_val) / overall_avg * 100) if overall_avg != 0 else 0
                    weaknesses.append(f"{col}: {entity_val:.2f} (-{diff:.1f}% below avg)")
            except:
                pass
        
        analysis += "STRENGTHS:\n" + "".join([f"  + {s}\n" for s in strengths[:5]])
        analysis += "\nWEAKNESSES:\n" + "".join([f"  - {w}\n" for w in weaknesses[:5]])
        analysis += "\nOPPORTUNITIES:\n  * Improve weak areas\n  * Leverage strengths\n"
        analysis += "\nTHREATS:\n  * Competition may match strengths\n  * Weaknesses may worsen\n"
    else:
        # List available entities so user knows what to ask for
        analysis += "Entity not found in dataset.\n"
        if cat_cols:
            for col in cat_cols[:3]:
                try:
                    unique_vals = df[col].dropna().unique()[:15]
                    string_vals = [str(v) for v in unique_vals if not str(v).replace('.','').replace('-','').isdigit()]
                    if string_vals:
                        analysis += f"Available in '{col}': {', '.join(string_vals[:10])}\n"
                        break
                except:
                    pass
        analysis += "\nShowing overall analysis.\n"
        # Show top performers as potential strengths
        if cat_cols and num_cols:
            analysis += "\nTOP PERFORMERS:\n"
            for col in num_cols[:3]:
                try:
                    top = df.nlargest(3, col)[[cat_cols[0], col]]
                    for _, row in top.iterrows():
                        analysis += f"  + {row[cat_cols[0]]}: {row[col]:.2f} ({col})\n"
                except:
                    pass
    return analysis + "\n"


def generate_risk_analysis(df, num_cols, cat_cols):
    """Generate risk matrix analysis"""
    analysis = "=== RISK ANALYSIS ===\n"
    risks = []
    
    for col in num_cols[:10]:
        try:
            mean_val, std_val, max_val = df[col].mean(), df[col].std(), df[col].max()
            cv = (std_val / mean_val * 100) if mean_val != 0 else 0
            extreme_ratio = (max_val / mean_val) if mean_val != 0 else 0
            
            if cv > 50 or extreme_ratio > 3:
                risks.append({
                    'factor': col, 'cv': cv,
                    'likelihood': "HIGH" if cv > 100 else "MEDIUM",
                    'impact': "HIGH" if extreme_ratio > 5 else "MEDIUM",
                    'score': cv * extreme_ratio
                })
        except:
            pass
    
    risks.sort(key=lambda x: -x['score'])
    analysis += "| Risk Factor | Likelihood | Impact | Score |\n|-------------|------------|--------|-------|\n"
    for r in risks[:10]:
        analysis += f"| {r['factor'][:30]} | {r['likelihood']} | {r['impact']} | {r['score']:.1f} |\n"
    return analysis + "\n"


def generate_benchmark(df, query, num_cols, cat_cols):
    """Generate benchmark comparison"""
    analysis = "=== BENCHMARK ANALYSIS ===\n"
    entity = extract_entity(df, query, cat_cols)
    
    if entity:
        analysis += f"Entity: {entity['name']} vs Average\n\n"
        analysis += "| Metric | Entity | Benchmark | Diff | Status |\n|--------|--------|-----------|------|--------|\n"
        
        for col in num_cols[:15]:
            try:
                entity_val = entity['data'][col].mean()
                benchmark = df[col].mean()
                diff_pct = ((entity_val - benchmark) / benchmark * 100) if benchmark != 0 else 0
                status = "ABOVE" if entity_val > benchmark else "BELOW"
                analysis += f"| {col[:25]} | {entity_val:.2f} | {benchmark:.2f} | {diff_pct:+.1f}% | {status} |\n"
            except:
                pass
    else:
        analysis += "OVERALL BENCHMARKS:\n"
        for col in num_cols[:10]:
            try:
                analysis += f"  {col}: avg={df[col].mean():.2f}, median={df[col].median():.2f}\n"
            except:
                pass
    return analysis + "\n"


def generate_outlier_analysis(df, num_cols, cat_cols):
    """Generate outlier detection"""
    analysis = "=== OUTLIER ANALYSIS ===\n"
    
    for col in num_cols[:5]:
        try:
            mean_val, std_val = df[col].mean(), df[col].std()
            upper, lower = mean_val + 2*std_val, mean_val - 2*std_val
            outliers = df[(df[col] > upper) | (df[col] < lower)]
            
            if len(outliers) > 0:
                analysis += f"\n{col}: Range [{lower:.2f}, {upper:.2f}], Outliers: {len(outliers)}\n"
                if cat_cols:
                    for _, row in outliers.head(5).iterrows():
                        analysis += f"  - {row[cat_cols[0]]}: {row[col]:.2f}\n"
        except:
            pass
    return analysis + "\n"


def generate_trend_analysis(df, num_cols, cat_cols):
    """Generate trend analysis"""
    analysis = "=== TREND ANALYSIS ===\n"
    
    for col in num_cols[:5]:
        try:
            values = df[col].dropna().values
            if len(values) > 10:
                first_half = values[:len(values)//2].mean()
                second_half = values[len(values)//2:].mean()
                change = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0
                direction = "INCREASING" if change > 5 else "DECREASING" if change < -5 else "STABLE"
                analysis += f"{col}: {direction} ({change:+.1f}%)\n"
        except:
            pass
    return analysis + "\n"


def generate_correlation(df, num_cols):
    """Generate correlation analysis"""
    analysis = "=== CORRELATION ANALYSIS ===\n"
    
    if len(num_cols) >= 2:
        try:
            corr_matrix = df[num_cols[:10]].corr()
            strong = []
            
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:
                        corr = corr_matrix.iloc[i, j]
                        if abs(corr) > 0.7:
                            strong.append((col1, col2, corr))
            
            for c in sorted(strong, key=lambda x: -abs(x[2]))[:10]:
                sign = "+" if c[2] > 0 else "-"
                analysis += f"  {c[0]} <-> {c[1]}: {sign}{abs(c[2]):.2f}\n"
        except:
            pass
    return analysis + "\n"


def generate_ranking(df, num_cols, cat_cols):
    """Generate ranking"""
    analysis = "=== RANKING ===\n"
    
    if cat_cols and num_cols:
        for col in num_cols[:3]:
            try:
                ranking = df.groupby(cat_cols[0])[col].mean().sort_values(ascending=False)
                analysis += f"\nTOP 10 BY {col}:\n"
                for i, (name, val) in enumerate(list(ranking.items())[:10], 1):
                    analysis += f"  {i}. {name}: {val:.2f}\n"
            except:
                pass
    return analysis + "\n"


def generate_pareto(df, num_cols, cat_cols):
    """Generate Pareto analysis"""
    analysis = "=== PARETO (80/20) ===\n"
    
    if cat_cols and num_cols:
        for col in num_cols[:2]:
            try:
                totals = df.groupby(cat_cols[0])[col].sum().sort_values(ascending=False)
                total_sum = totals.sum()
                cumsum, count_80 = 0, 0
                
                for val in totals.values:
                    cumsum += val
                    count_80 += 1
                    if cumsum >= total_sum * 0.8:
                        break
                
                pct = count_80 / len(totals) * 100
                analysis += f"{col}: {count_80} entities ({pct:.1f}%) = 80% of total\n"
            except:
                pass
    return analysis + "\n"


def generate_injury_risk(df, num_cols, cat_cols):
    """Generate injury/risk indicator analysis - DYNAMIC"""
    analysis = "=== HIGH VARIABILITY RISK INDICATORS ===\n"
    
    # Find columns with high variability (potential risk indicators)
    risk_cols = []
    for col in num_cols:
        try:
            cv = df[col].std() / df[col].mean() * 100 if df[col].mean() != 0 else 0
            if cv > 50:  # High coefficient of variation
                risk_cols.append((col, cv))
        except:
            pass
    
    risk_cols.sort(key=lambda x: -x[1])
    
    for col, cv in risk_cols[:5]:
        try:
            threshold = df[col].mean() + 1.5 * df[col].std()
            high_risk = df[df[col] > threshold]
            
            analysis += f"\n{col} (CV={cv:.1f}%, threshold={threshold:.2f}):\n"
            if len(high_risk) > 0 and cat_cols:
                for _, row in high_risk.head(5).iterrows():
                    analysis += f"  ! {row[cat_cols[0]]}: {row[col]:.2f}\n"
        except:
            pass
    return analysis + "\n"


def is_id_column(col_name, series, n_samples):
    """Check if column is an ID/index - should be excluded from stats"""
    col_lower = col_name.lower()
    # Name-based detection
    is_id_name = any(x in col_lower for x in ['_id', 'id_', '.id', 'index', '_key', 'key_'])
    is_id_name = is_id_name or col_lower.endswith('id') or col_lower == 'id'
    # Value-based detection - high cardinality, sequential
    if not is_id_name and len(series) > 0:
        nunique = series.nunique()
        if nunique > n_samples * 0.8:  # Very high cardinality
            is_id_name = True
    return is_id_name

def generate_analytics(df, query, detected_types):
    """Main analytics generator"""
    analysis = ""
    # Filter out ID columns from numeric columns
    all_num_cols = df.select_dtypes(include=['number']).columns.tolist()
    num_cols = [col for col in all_num_cols if not is_id_column(col, df[col], len(df))]
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(all_num_cols) != len(num_cols):
        excluded = set(all_num_cols) - set(num_cols)
        print(f"Analytics: Excluded ID columns from stats: {list(excluded)[:5]}")
    
    for detection in detected_types[:2]:
        atype = detection['type']
        
        if atype == 'swot':
            analysis += generate_swot(df, query, num_cols, cat_cols)
        elif atype in ['risk_matrix', 'risk_register', 'hazard', 'threat', 'risk_appetite', 'injury_risk']:
            analysis += generate_risk_analysis(df, num_cols, cat_cols)
            analysis += generate_injury_risk(df, num_cols, cat_cols)
        elif atype == 'benchmark':
            analysis += generate_benchmark(df, query, num_cols, cat_cols)
        elif atype == 'outlier':
            analysis += generate_outlier_analysis(df, num_cols, cat_cols)
        elif atype == 'trend':
            analysis += generate_trend_analysis(df, num_cols, cat_cols)
        elif atype == 'correlation':
            analysis += generate_correlation(df, num_cols)
        elif atype in ['ranking', 'league']:
            analysis += generate_ranking(df, num_cols, cat_cols)
        elif atype == 'pareto':
            analysis += generate_pareto(df, num_cols, cat_cols)
        elif atype in ['comparison', 'peer', 'competitive']:
            analysis += generate_benchmark(df, query, num_cols, cat_cols)
        else:
            # DYNAMIC QUERY-AWARE ANALYSIS
            analysis += f"=== {atype.upper()} ANALYSIS ===\n"
            analysis += f"Dataset: {len(df)} rows, {len(df.columns)} columns\n\n"
            
            # Extract keywords from query
            query_words = [w.lower() for w in re.sub(r'[^a-zA-Z0-9\s]', '', query).split() if len(w) > 2]
            
            # Find matching columns based on query
            matched_num = []
            matched_cat = []
            # Score-based matching for numeric columns
            scored_num = []
            for col in num_cols:
                col_lower = col.lower()
                score = 0
                for w in query_words:
                    if w in col_lower:
                        score += len(w) * 2
                        if col_lower.startswith(w) or col_lower.endswith(w):
                            score += 10
                    if col_lower in w:
                        score += len(col_lower)
                if score > 0:
                    scored_num.append((col, score))
            matched_num = [c[0] for c in sorted(scored_num, key=lambda x: -x[1])]
            # Priority: name columns first, then other matches
            name_cols = [c for c in cat_cols if 'name' in c.lower()]
            for col in name_cols:
                col_lower = col.lower()
                for w in query_words:
                    if w in col_lower:
                        matched_cat.insert(0, col)  # High priority
                        break
                else:
                    matched_cat.append(col)  # Name cols are always useful
            
            for col in cat_cols:
                if col in matched_cat:
                    continue
                col_lower = col.lower()
                for w in query_words:
                    if w in col_lower:
                        matched_cat.append(col)
                        break
            
            # Use matched columns or fallback to first ones
            target_num = matched_num[:5] if matched_num else num_cols[:5]
            target_cat = matched_cat[:2] if matched_cat else [c for c in cat_cols if 'name' in c.lower()][:2]
            if not target_cat:
                target_cat = cat_cols[:2]
            
            # Generate grouped analysis if we have both
            if target_num and target_cat:
                for num_col in target_num[:3]:
                    for cat_col in target_cat[:1]:
                        try:
                            # Determine aggregation from query
                            if any(w in query for w in ['average', 'avg', 'mean']):
                                result = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
                                agg_name = "AVERAGE"
                            elif any(w in query for w in ['max', 'highest', 'top', 'best']):
                                result = df.groupby(cat_col)[num_col].max().sort_values(ascending=False)
                                agg_name = "MAX"
                            elif any(w in query for w in ['min', 'lowest', 'least']):
                                result = df.groupby(cat_col)[num_col].min().sort_values(ascending=True)
                                agg_name = "MIN"
                            elif any(w in query for w in ['count', 'how many']):
                                result = df.groupby(cat_col)[num_col].count().sort_values(ascending=False)
                                agg_name = "COUNT"
                            else:
                                result = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
                                agg_name = "TOTAL"
                            
                            analysis += f"{agg_name} {num_col} BY {cat_col}:\n"
                            for idx, val in list(result.items())[:15]:
                                analysis += f"  {idx}: {val:.2f}\n"
                            analysis += "\n"
                        except Exception as e:
                            pass
            elif target_num:
                for col in target_num[:5]:
                    try:
                        analysis += f"{col}: total={df[col].sum():.2f}, avg={df[col].mean():.2f}, max={df[col].max():.2f}, min={df[col].min():.2f}\n"
                    except:
                        pass
            analysis += "\n"
    
    return analysis
