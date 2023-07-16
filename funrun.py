import os
import random
from datetime import datetime

def create_run_name(nouns_file, adjectives_file):
    # Read in the nouns and adjectives
    with open(nouns_file, 'r') as f:
        nouns = [line.strip() for line in f.readlines()]
    with open(adjectives_file, 'r') as f:
        adjectives = [line.strip() for line in f.readlines()]

    # Randomly select a noun and adjective
    noun = random.choice(nouns)
    adjective = random.choice(adjectives)

    # Get current date and time
    now = datetime.now()

    # Format as a string
    now_str = now.strftime("%Y%m%d-%H%M%S")

    # Construct run name
    run_name = f"lipsync_{adjective}_{noun}_{now_str}"

    return run_name
