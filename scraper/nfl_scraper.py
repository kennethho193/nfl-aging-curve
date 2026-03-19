import nfl_data_py as nfl
import pandas as pd
import os

def scrape_rb_stats(start_year, end_year):
    """
    Scrapes seasonal rushing stats and merges with player info to get position and age for aging curve analysis.
    """
    years = list(range(start_year, end_year + 1))

    #Step 1: Get seasonal stats
    print("Getting seasonal stats:")
    stats_df = nfl.import_seasonal_data(years)
    print(f"  Got {len(stats_df)} rows")

    #Step 2: Get player info
    print("Getting player info:")
    players_df = nfl.import_players()
    print(f"  Got {len(players_df)} players")

    #Step 3: Keep only useful player info columns
    players_slim = players_df[[
        "gsis_id", "display_name", "position",
        "birth_date", "height", "weight",
        "college_name", "draft_year", "draft_round"
    ]].copy()

    #Step 4: Merge stats with player info
    print("Merging datasets:")
    merged_df = stats_df.merge(
        players_slim,
        left_on="player_id",
        right_on="gsis_id",
        how="inner"
    )
    print(f"  Merged dataset has {len(merged_df)} rows")

    #Step 5: Filter to RBs only
    rb_df = merged_df[merged_df["position"] == "RB"].copy()
    print(f"  Filtered to RBs — {len(rb_df)} rows")

    #Step 6: Calculate age at each season
    rb_df["birth_date"] = pd.to_datetime(rb_df["birth_date"])
    rb_df["age"] = rb_df["season"] - rb_df["birth_date"].dt.year
    print(f"  Age column calculated")

    #Step 7: Filter to meaningful sample — 50+ carries in a season
    rb_df = rb_df[rb_df["carries"] >= 50].copy()
    print(f"  Filtered to 50+ carries — {len(rb_df)} rows")

    #Step 8: Keep only the columns we need
    cols = [
        "player_id", "display_name", "season", "age",
        "position", "college_name", "draft_year", "draft_round",
        "height", "weight", "games",
        "carries", "rushing_yards", "rushing_tds",
        "rushing_yards_after_contact", "rushing_first_downs",
        "rushing_epa", "rushing_fumbles",
        "receptions", "targets", "receiving_yards",
        "receiving_tds", "receiving_epa",
        "fantasy_points", "fantasy_points_ppr"
    ]

    available_cols = [c for c in cols if c in rb_df.columns]
    rb_df = rb_df[available_cols]

    return rb_df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    df = scrape_rb_stats(2000, 2023)

    output_path = "data/raw/rb_rushing_stats.csv"
    df.to_csv(output_path, index=False)

    print(f"\nDone! Saved {len(df)} rows to {output_path}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nAge range in dataset:")
    print(f"  Min age: {df['age'].min()}")
    print(f"  Max age: {df['age'].max()}")
    print(f"  Most common ages: {df['age'].value_counts().head(5).to_dict()}")