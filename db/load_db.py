import sqlite3
import pandas as pd
import os

def create_database(db_path, schema_path):
    """
    Creates the SQLite database from the schema file.
    """

    print("Creating database from schema:")

    with open(schema_path, "r") as f:
        schema = f.read()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(schema)
    conn.commit()
    conn.close()

    print(" Database and tables created successfully.")

def load_players(conn, df):
    """
    Loads unique player data into the players.
    """

    print("Loading players data:")

    players_df = df[[
        "player_id", "display_name", "position",
        "college_name", "draft_year", "draft_round",
        "height", "weight"
    ]].drop_duplicates(subset=["player_id"])

    players_df.to_sql(
        "players",
        conn,
        if_exists="append",
        index=False
    )

    print(f"    Loaded {len(players_df)} unique players.")

def load_season_stats(conn, df):
    """
    Loads season-level stats into the season_stats table.
    """
    stats_df = df[[
        "player_id", "season", "age", "games",
        "carries", "rushing_yards", "rushing_tds",
        "rushing_first_downs",
        "rushing_epa", "rushing_fumbles",
        "receptions", "targets", "receiving_yards",
        "receiving_tds", "receiving_epa",
        "fantasy_points", "fantasy_points_ppr"
    ]].copy()


    stats_df.to_sql(
        "season_stats",
        conn,
        if_exists="append",
        index=False
    )

    print(f"  Loaded {len(stats_df)} season records")

def verify_load(conn):
    """
    Runs a quick sanity check on the loaded data.
    """
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM players")
    player_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM season_stats")
    stats_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT p.display_name, s.season, s.rushing_yards
        FROM season_stats s
        JOIN players p ON s.player_id = p.player_id
        ORDER BY s.rushing_yards DESC
        LIMIT 5
    """)
    top_seasons = cursor.fetchall()

    print(f"\nVerification:")
    print(f"  Players in database: {player_count}")
    print(f"  Season records in database: {stats_count}")
    print(f"\n  Top 5 rushing seasons in dataset:")
    for name, season, yards in top_seasons:
        print(f"    {name} ({season}): {yards} yards")

if __name__ == "__main__":
    db_path     = "db/nfl_aging.db"
    schema_path = "db/schema.sql"
    csv_path    = "data/raw/rb_rushing_stats.csv"

    #Step 1: Create database
    create_database(db_path, schema_path)

    #Step 2: Load CSV
    print("\nLoading CSV...")
    df = pd.read_csv(csv_path)
    print(f"  Read {len(df)} rows from CSV")

    #Step 3: Connect and load
    conn = sqlite3.connect(db_path)

    print("\nLoading players table:")
    load_players(conn, df)

    print("\nLoading season_stats table:")
    load_season_stats(conn, df)

    conn.commit()
    conn.close()

    #Step 4: Verify
    conn = sqlite3.connect(db_path)
    verify_load(conn)
    conn.close()

    print("\nDone! Database saved to db/nfl_aging.db")