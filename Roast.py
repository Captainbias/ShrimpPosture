import random
from pync import Notifier

def load_roasts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.startswith('"') and line.endswith('\n')]
    return lines

def random_roast(roasts):
    return random.choice(roasts)

def alert_bad_posture(message):
    Notifier.notify(message, title="Posture Alert")

if __name__ == "__main__":
    file_path = "PostureRoast.txt"  # Make sure the file is in the same directory
    try:
        roasts = load_roasts(file_path)
        print("🔥 POSTURE ROAST MODE ACTIVATED 🔥")
        print(random_roast(roasts))
        alert_bad_posture(random_roast(roasts))
    except FileNotFoundError:
        print("Oops! Make sure 'PostureRoast.txt' is in the same directory as this script.")