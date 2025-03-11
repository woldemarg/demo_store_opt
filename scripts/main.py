import streamlit as st
import pulp
import pandas as pd
import numpy as np


# %%

def generate_synthetic_data():

    num_categories = np.random.randint(10, 16)
    categories = [f"Category_{i:02d}" for i in range(1, num_categories + 1)]
    data = []

    for category in categories:
        num_priorities = np.random.randint(1, 3)
        for priority in range(1, num_priorities + 1):
            num_series = np.random.randint(1, 3)
            # Convert series to letters (A, B, C...)
            series_labels = [chr(65 + i) for i in range(num_series)]

            for series in series_labels:
                num_blocks = np.random.randint(5, 8)
                for _ in range(num_blocks):
                    area = round(np.random.uniform(2, 15), 2)

                    # Non-linear decreasing function for marginality based on area

                    noise = np.random.normal(0, 0.03)

                    # marginality = round(np.exp(-area / 7), 2) + noise
                    marginality = round(
                        1 / (1 + np.exp(-area / 7)), 2) + noise

                    # Ensure the marginality stays within [0, 1] after adding noise
                    marginality = np.clip(marginality, 0, 1)

                    data.append(
                        [category, priority, series, area, marginality])

    df = pd.DataFrame(
        data, columns=["category", "priority", "series", "area", "marginality"])

    # Sorting: category (asc), priority (asc), series (asc), area (desc)
    df.sort_values(by=["category", "priority", "series", "area"], inplace=True)

    # Assign block labels: "Category_01_B01"
    df.insert(1, 'block_id', df.groupby("category").cumcount() + 1)

    df["block_id"] = df.apply(
        lambda row: f"{row['category']}_B{row['block_id']:02d}", axis=1)

    return df.reset_index(drop=True)


def select_blocks(blocks):

    first_priority = (blocks[blocks['priority'] == 1]
                      .groupby(['category', 'series'])
                      # .groupby('category')
                      .sample(1))

    extra_categories = np.random.choice(
        blocks['category'].unique(),
        np.random.randint(0, len(blocks['category'].unique()) // 2),
        replace=False)

    extra_blocks = (
        blocks[(blocks['category'].isin(extra_categories)) &
               (blocks['priority'] > 1)]
        .groupby(['category', 'series'], group_keys=False)
        .sample(1))

    return pd.concat([first_priority, extra_blocks]).reset_index(drop=True)


def format_optimized_results(blocks, selected_block_ids):

    optimized_blocks = blocks[blocks['block_id'].isin(
        selected_block_ids)].copy()

    optimized_blocks['margin'] = optimized_blocks['area'] * \
        optimized_blocks['marginality']

    return optimized_blocks.sort_values(['category', 'priority', 'series']).reset_index(drop=True)


def optimize_data(blocks, required_area):

    # Create the optimization problem
    prob = pulp.LpProblem("Maximize_Store_Margin", pulp.LpMaximize)

    # Create decision variables with more descriptive names
    is_block_selected = {
        b: pulp.LpVariable(f"select_block_{b}", cat="Binary")
        for b in blocks['block_id']
    }

    is_category_used = {
        cat: pulp.LpVariable(f"use_category_{cat}", cat="Binary")
        for cat in required_area
    }

    # Area deviation tracking variables
    area_deviation = {
        cat: (
            pulp.LpVariable(f"excess_area_{cat}", lowBound=0),
            pulp.LpVariable(f"deficit_area_{cat}", lowBound=0)
        )
        for cat in required_area
    }

    # Priority tracking variables
    is_priority_used = {
        (cat, pri): pulp.LpVariable(f"use_priority_{cat}_{pri}", cat="Binary")
        for cat in required_area
        for pri in blocks[blocks['category'] == cat]['priority'].unique()
    }

    # Precompute block margins and cache for better performance
    blocks['margin'] = blocks['area'] * blocks['marginality']

    # Get maximum priority for penalty calculation
    max_priority = blocks['priority'].max()

    # Build objective function components
    # 1. Margin maximization term
    margin_terms = [
        is_block_selected[b] *
        blocks.loc[blocks['block_id'] == b, 'margin'].values[0]
        for b in is_block_selected
    ]

    # 2. Priority penalty term - prefer higher priority blocks
    priority_penalty_terms = []
    for cat in required_area:
        cat_priorities = sorted(
            blocks[blocks['category'] == cat]['priority'].unique(), reverse=True)
        for pri in cat_priorities:
            penalty = max_priority - pri + 1  # Higher priority gets lower penalty
            pri_blocks = blocks[
                (blocks['category'] == cat) &
                (blocks['priority'] == pri)
            ]['block_id'].tolist()

            priority_penalty_terms.extend(
                [penalty * is_block_selected[b] for b in pri_blocks])

    # 3. Area deviation penalty terms
    excess_area_penalty = [area_deviation[cat][0] for cat in required_area]
    deficit_area_penalty = [area_deviation[cat][1] for cat in required_area]

    # Penalty weights
    PRIORITY_WEIGHT = 0.05      # Weight for priority penalties
    EXCESS_AREA_WEIGHT = 2.5   # Higher penalty for exceeding area
    DEFICIT_AREA_WEIGHT = 0.5  # Lower penalty for under-allocating area

    # Complete objective function
    prob += (
        pulp.lpSum(margin_terms) -
        PRIORITY_WEIGHT * pulp.lpSum(priority_penalty_terms) -
        EXCESS_AREA_WEIGHT * pulp.lpSum(excess_area_penalty) -
        DEFICIT_AREA_WEIGHT * pulp.lpSum(deficit_area_penalty)
    )

    # Add constraints grouped by purpose
    # 1. Category area constraints
    for cat, req_area in required_area.items():
        cat_blocks = blocks[blocks['category'] == cat]['block_id'].tolist()
        excess_area, deficit_area = area_deviation[cat]

        # Skip categories with zero required area
        if req_area == 0:
            prob += pulp.lpSum(is_block_selected[b] for b in cat_blocks) == 0
            continue

        # Category must be used if area requirement > 0
        prob += is_category_used[cat] == 1

        # At least one block must be selected if category is used
        prob += pulp.lpSum(is_block_selected[b]
                           for b in cat_blocks) >= is_category_used[cat]

        # Area deviation constraint
        total_area = pulp.lpSum(
            is_block_selected[b] *
            blocks.loc[blocks['block_id'] == b, 'area'].values[0]
            for b in cat_blocks
        )
        prob += total_area - req_area == excess_area - deficit_area

    # 2. Priority hierarchy constraints
    for cat in required_area:
        for pri in blocks[blocks['category'] == cat]['priority'].unique():
            pri_blocks = blocks[
                (blocks['category'] == cat) &
                (blocks['priority'] == pri)
            ]['block_id'].tolist()

            # Link priority usage variable to block selection
            prob += pulp.lpSum([is_block_selected[b] for b in pri_blocks]
                               ) <= len(pri_blocks) * is_priority_used[(cat, pri)]
            prob += pulp.lpSum([is_block_selected[b]
                               for b in pri_blocks]) >= is_priority_used[(cat, pri)]

            # Priority order constraint - can only use lower priority if higher priority is exhausted
            if pri > 1 and (pri - 1) in blocks[blocks['category'] == cat]['priority'].unique():
                prob += is_priority_used[(cat, pri)
                                         ] <= is_priority_used[(cat, pri - 1)]

    # 3. Series incompatibility constraints
    # for cat in required_area:
    #     for priority in blocks[blocks['category'] == cat]['priority'].unique():
    #         # Group blocks by series
    #         series_groups = blocks[
    #             (blocks['category'] == cat) &
    #             (blocks['priority'] == priority)
    #         ].groupby('series')

    #         for series, group in series_groups:
    #             incompatible_blocks = group['block_id'].tolist()

    #             if len(incompatible_blocks) > 1:
    #                 # At most one block from the same series can be selected
    #                 prob += pulp.lpSum(is_block_selected[b]
    #                                    for b in incompatible_blocks) <= 1

    # Remove the old series incompatibility constraints (section 3)
    # Replace with new series selection constraints

    # 3. Series selection constraints
    for cat in required_area:
        for priority in blocks[blocks['category'] == cat]['priority'].unique():
            # Group blocks by series
            series_in_priority = blocks[
                (blocks['category'] == cat) &
                (blocks['priority'] == priority)
            ]['series'].unique()

            # Create series usage variables
            is_series_used = {}
            for series in series_in_priority:
                is_series_used[series] = pulp.LpVariable(
                    f"use_series_{cat}_{priority}_{series}", cat="Binary")

                # Get all blocks in this series
                series_blocks = blocks[
                    (blocks['category'] == cat) &
                    (blocks['priority'] == priority) &
                    (blocks['series'] == series)
                ]['block_id'].tolist()

                # Link series usage variable to block selection
                # If any block from the series is selected, the series is used
                prob += pulp.lpSum([is_block_selected[b] for b in series_blocks]
                                   ) <= len(series_blocks) * is_series_used[series]
                prob += pulp.lpSum([is_block_selected[b]
                                   for b in series_blocks]) >= is_series_used[series]

                # For each series, still ensure at most one block is selected
                prob += pulp.lpSum(is_block_selected[b]
                                   for b in series_blocks) <= 1

            # Ensure each series in this priority is used if the priority is used
            # But only if using all series doesn't exceed area requirements too much
            if len(series_in_priority) > 0:
                prob += pulp.lpSum([is_series_used[series] for series in series_in_priority]) >= \
                    is_priority_used[(cat, priority)] * len(series_in_priority)

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=180))

    selected_blocks = [b for b in is_block_selected if pulp.value(
        is_block_selected[b]) == 1]

    results = {
        'status': pulp.LpStatus[prob.status],
        'assigned_blocks': [],
        'category_summary': {},
        'total_margin': 0,
        'total_area': 0,
        'total_required_area': sum(required_area.values()),
        'total_positive_delta': 0,
        'total_negative_delta': 0
    }

    # Process assigned blocks
    for b in is_block_selected:
        if pulp.value(is_block_selected[b]) == 1:
            block_data = blocks[blocks['block_id'] == b].iloc[0].to_dict()
            block_data['margin'] = block_data['area'] * \
                block_data['marginality']
            results['assigned_blocks'].append(block_data)
            results['total_margin'] += block_data['margin']
            results['total_area'] += block_data['area']

    # Summarize by category
    for cat in required_area:
        if required_area[cat] > 0:
            cat_blocks = [b for b in results['assigned_blocks']
                          if b['category'] == cat]
            assigned_area = sum(b['area'] for b in cat_blocks)
            delta = assigned_area - required_area[cat]

            # Track total positive and negative deltas
            if delta > 0:
                results['total_positive_delta'] += delta
            else:
                results['total_negative_delta'] += abs(delta)

            results['category_summary'][cat] = {
                'required': required_area[cat],
                'assigned': assigned_area,
                'delta': delta,
                'delta_percentage': (delta / required_area[cat] * 100) if required_area[cat] > 0 else 0,
                'blocks': len(cat_blocks),
                'priority_breakdown': {}
            }

            # Add priority breakdown
            for pri in sorted(blocks[blocks['category'] == cat]['priority'].unique()):
                pri_blocks = [b for b in cat_blocks if b['priority'] == pri]
                if pri_blocks:
                    results['category_summary'][cat]['priority_breakdown'][pri] = {
                        'blocks': len(pri_blocks),
                        'area': sum(b['area'] for b in pri_blocks),
                        'margin': sum(b['area'] * b['marginality'] for b in pri_blocks)
                    }

    # Format the optimized results for display
    return format_optimized_results(blocks, selected_blocks), results


def compare_store_layouts(initial_df, optimized_df):
    """
    Create a side-by-side comparison of initial and optimized store layouts.

    Args:
        initial_df (DataFrame): Initial block selection
        optimized_df (DataFrame): Optimized block selection

    Returns:
        DataFrame: Merged comparison showing both layouts by category with detailed block info
    """
    # Process initial blocks with per-block details
    initial_blocks_detail = {}
    initial_area_total = 0
    initial_margin_total = 0

    for category in initial_df['category'].unique():
        cat_blocks = initial_df[initial_df['category'] == category]
        block_details = []

        for _, row in cat_blocks.iterrows():
            block_details.append(
                f"{row['block_id']} (A:{row['area']:.1f}, M:{row['margin']:.1f})"
            )
            initial_area_total += row['area']
            initial_margin_total += row['margin']

        initial_blocks_detail[category] = {
            'blocks': ' + '.join(block_details),
            'area': cat_blocks['area'].sum(),
            'margin': cat_blocks['margin'].sum()
        }

    # Process optimized blocks with per-block details
    optimized_blocks_detail = {}
    optimized_area_total = 0
    optimized_margin_total = 0

    for category in optimized_df['category'].unique():
        cat_blocks = optimized_df[optimized_df['category'] == category]
        block_details = []

        for _, row in cat_blocks.iterrows():
            block_details.append(
                f"{row['block_id']} (A:{row['area']:.1f}, M:{row['margin']:.1f})"
            )
            optimized_area_total += row['area']
            optimized_margin_total += row['margin']

        optimized_blocks_detail[category] = {
            'blocks': ' + '.join(block_details),
            'area': cat_blocks['area'].sum(),
            'margin': cat_blocks['margin'].sum()
        }

    # Create a full set of categories from both dataframes
    all_categories = sorted(set(list(initial_blocks_detail.keys()) +
                                list(optimized_blocks_detail.keys())))

    # Build the comparison DataFrame
    comparison_data = []

    for category in all_categories:
        init_data = initial_blocks_detail.get(
            category, {'blocks': '', 'area': 0, 'margin': 0})
        opt_data = optimized_blocks_detail.get(
            category, {'blocks': '', 'area': 0, 'margin': 0})

        area_diff = opt_data['area'] - init_data['area']
        margin_diff = opt_data['margin'] - init_data['margin']

        comparison_data.append({
            'category': category,
            'ini_blocks': init_data['blocks'],
            'ini_area': init_data['area'],
            'ini_margin': init_data['margin'],
            'opt_blocks': opt_data['blocks'],
            'opt_area': opt_data['area'],
            'opt_margin': opt_data['margin'],
            'area_diff': area_diff,
            'margin_diff': margin_diff
        })

    # Create DataFrame from the comparison data
    comparison_df = pd.DataFrame(comparison_data)

    # Function to format the '_diff' columns with + for positive and - for negative differences
    def format_diff(val):
        return f"{'+' if val > 0 else ''}{val:.1f}"

    # Apply the formatting to 'area_diff' and 'margin_diff' columns
    comparison_df['area_diff'] = comparison_df['area_diff'].apply(format_diff)
    comparison_df['margin_diff'] = comparison_df['margin_diff'].apply(
        format_diff)

    area_total_diff = optimized_area_total - initial_area_total
    margin_total_diff = optimized_margin_total - initial_margin_total

    area_pct_change = (area_total_diff / initial_area_total *
                       100) if initial_area_total else 0
    margin_pct_change = (
        margin_total_diff / initial_margin_total * 100) if initial_margin_total else 0

    # Add totals row with percentage changes
    row = pd.DataFrame({
        'category': 'TOTAL',
        'ini_blocks': '',
        'ini_area': initial_area_total,
        'ini_margin': initial_margin_total,
        'opt_blocks': '',
        'opt_area': optimized_area_total,
        'opt_margin': optimized_margin_total,
        'area_diff': f"{area_total_diff:+.1f} ({area_pct_change:+.1f}%)",
        'margin_diff': f"{margin_total_diff:+.1f} ({margin_pct_change:+.1f}%)"
    },
        index=[0])

    # Add totals row
    comparison_df = pd.concat([comparison_df, row], ignore_index=True)

    return comparison_df.round(1)

# %%


st.title("Cистема оптимізації: опис")

st.markdown("""
## 1. Загальна мета
Максимізувати дохід магазину, оптимально розподіляючи доступні блоки (кластери) між товарними категоріями.
## 2. Правила оптимізації
### 2.1. Вимоги покриття площі
Кожна категорія має визначену площу, яку необхідно покрити кластерами (вхідні обмеження / параметри магазину). Виділена площа (фактична площа кластерів) повинна якомога точніше відповідати цій потребі, допускаючи незначні відхилення. Загальна площа всіх призначених кластерів під категоріями має бути максимально наближеною до сумарної необхідної площі.
### 2.2. Пріоритети і серії кластерів
Кластери мають пріоритети — перевага надається тим, що мають вищий рівень (задається у вхідному довіднику / параметрах кластерів). Усередині пріоритету одного рівня кластери групуються в лінійки (серії). В межах кожного пріоритету (під відповідну категорію) повинні бути представлені кластери з усіх серій, дотримуючись правила: одна серія — тільки один блок.
## 3. Генерація довідника кластерів
### 3.1. Загальні правила генерації довідника:
- довідник генерується для 10-15 товарних категорій
- маржинальність блоків (дохід на кв. м) задається нелінійною зростаючою функцією, тобто більша площа може приносити пропорційно вищий дохід
- кількість пріоритетів всередині категорії і кількість лінійок (серій) визначається випадково, кількість блоків в серії 5-7 шт.
""")

container = st.container()
# Generate and display synthetic data button
if st.button("Згенерувати довідник"):
    blocks = generate_synthetic_data()
    st.session_state.blocks = blocks.copy()

    with container:
        st.markdown("### 3.2. Характеристики / параметри кластерів")
        st.dataframe(blocks)

st.markdown("""
##  4. Генерація магазину
### 4.1. Загальні правила генерації магазину:
- для кожної категорії із довідника обираються лінійки (серії) блоків із найвищим пріоритетом, із кожної такої серії випадково обирається один блок
- деяким категоріям (випадково) можуть додаватися блоки із меншим пріоритетом
""")
container_store = st.container()
container_dict = st.container()
container_opt = st.container()
# Button to run optimization
if st.button("Згенерувати і оптимізувати магазин"):

    required = (select_blocks(st.session_state.blocks)
                .assign(margin=lambda x: x['area'] * x['marginality'])
                .sort_values(['category', 'priority', 'series'])
                .reset_index(drop=True))

    st.session_state.required = required

    if 'blocks' in st.session_state:
        blocks = st.session_state.blocks.copy()
        required = st.session_state.required.copy()

        with container:
            st.markdown("### 3.2. Характеристики / параметри кластерів")
            st.dataframe(st.session_state.blocks)

        with container_store:
            st.markdown("### 4.2. Кластери у магазині")
            st.dataframe(st.session_state.required)

        required_area = (required
                         .groupby('category')[['area', 'margin']]
                         .sum())

        init_margin = required_area['margin'].sum()

        required_area = required_area['area'].to_dict()

        st.session_state.required_area = required_area

        with container_dict:
            st.markdown('''
            ## 5. Механізм оптимізації
            ### 5.1. Вхідні площі для покриття
            ''')
            st.json(st.session_state.required_area)

        optimized_result, results = optimize_data(blocks, required_area)
        # Compare initial and optimized data
        comparison_result = compare_store_layouts(
            required, optimized_result)

        summary = {
            'Total Required Area': f"{results['total_required_area']:.2f}",
            'Total Assigned Area': f"{results['total_area']:.2f}",
            'Total Area Delta': f"{results['total_area'] - results['total_required_area']:.2f} "
            f"({(results['total_area'] - results['total_required_area']) / results['total_required_area'] * 100:.2f}%)",
            'Total Positive Delta (Over-allocation)': f"{results['total_positive_delta']:.2f}",
            'Total Negative Delta (Under-allocation)': f"{results['total_negative_delta']:.2f}",
            'Total Init Margin': f'{init_margin:.2f}',
            'Total Opti Margin': f"{results['total_margin']:.2f}", }

        with container_dict:
            st.markdown('### 5.2. Загальні результати')
            st.json(summary)
            st.markdown("### 5.3. Деталі оптимізації")
            st.dataframe(comparison_result)

    else:
        st.error("Спочатку треба згенерувати довідник блоків")
