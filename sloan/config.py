"""Every threshold from the walkthrough document lives here, so a change of
definition is a one-line edit and not a hunt through the pipeline."""

# ---- Collecting Games -------------------------------------------------------
ARCHIVE_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
OPENINGS_URL = "https://raw.githubusercontent.com/lichess-org/chess-openings/master/{letter}.tsv"
GAME_TYPES = ["bullet", "blitz", "rapid"]

# Upper bounds (inclusive) of elo classes 1..7; class 8 is n > 2600.
ELO_CLASS_UPPER = [1000, 1250, 1500, 1750, 2000, 2300, 2600]
# class -> level: 1 low, 2-4 medium, 5-7 high, 8 top
ELO_LEVELS = ["low", "medium", "high", "top"]
CLASS_TO_LEVEL = {1: "low", 2: "medium", 3: "medium", 4: "medium",
                  5: "high", 6: "high", 7: "high", 8: "top"}
# A class counts as unbalanced if its game count is off the expected count by more than this.
BALANCE_TOLERANCE = 0.25

# ---- Classification within --------------------------------------------------
EVAL_CAP = 2000            # evaluations clamped to +-2000cp; a forced mate counts as +-2000
CPL_CAP = 2000             # "self cap cpl at 2000"
WIN_K = 0.00368208         # Lichess win% constant
ENDGAME_MATERIAL = 20      # endgame = fewer than 20 points of non-pawn material (both sides)
TIME_PRESSURE_SECONDS = 15  # no-increment games: moves made with <= 15s left are dropped
PIECE_POINTS = {2: 3, 3: 3, 4: 5, 5: 9}  # knight, bishop, rook, queen

# ---- Collecting Moves | Shocks ---------------------------------------------
MISTAKE_CP = 100           # cpl >= 100 -> mistake / potential shock
CAPITALIZED_CP = 100       # still >= 100cp worse after the opponent's reply -> real shock
PARTIAL_GAP_CP = 50        # gap >= 50cp between loss and what was kept -> "not fully capitalized"
PERSIST_PLIES = 6          # the loss must still be there 6 plies later -> "significant impact"
MERGE_PLIES = 4            # shock within p+2 / p+4 of the same player's shock -> repercussion

# ---- Eva / new data ---------------------------------------------------------
DECIDED_HIGH, DECIDED_LOW = 95.0, 5.0
WINDOW_MOVES = 5           # pre / post windows: own moves p-+2 ... p-+10
WINDOW_MIN_VALID = 3       # windows with fewer usable moves than this are left empty
RECOVERY_MOVES = 10        # post-20-plies sequence: own moves p+2 ... p+20
AHEAD, BEHIND = 55.0, 45.0
WPL_PILE_EDGES = [5, 10, 15, 20, 25, 30, 35]             # upper bounds of piles 1..7
CPL_PILE_EDGES = [100, 200, 300, 400, 500, 600, 700]
PILE_MIN_EVENTS = 30       # an elo level with fewer events than this reuses the global cut-offs
SWIS_PILE = "impact_pile_cpl"   # which impact score weights "swis"
STRESS_MIN_CLOCK = 0.5     # floor for the clock (seconds) so stress never divides by zero

# ---- Analysis from data -----------------------------------------------------
MIN_POINT_N = 20           # graph points backed by fewer events than this are not drawn
# "level-level" is not analysed: a >=100cp loss almost never leaves a level position level
CONDITIONS = ["ahead-ahead", "ahead-level", "level-behind", "ahead-behind", "behind-behind"]
STRESS_LEVELS = ["low", "medium", "high"]
TEMPO_LEVELS = ["impulsive", "normal", "cognitive"]
FAVOR_LEVELS = ["favor-player", "back-and-forth", "favor-opponent"]
CATEGORIES = ["mistake", "shock_nonsig", "shock_sig"]
CATEGORY_LABELS = {"mistake": "Mistakes (not capitalized)",
                   "shock_nonsig": "Shocks, no lasting impact",
                   "shock_sig": "Shocks, lasting impact",
                   "all": "All mistakes and shocks"}
