import argparse
import json



def generate_sitrep(badge_label, tags, digest, outcome, badge_filename_full):
    """
    Function to generate the badge sitrep 

    Args: 
        badge_label (str): The label of the badge
        tags (str): The tags of the badge
        digest (str): The digest of the badge
        outcome (str): The outcome of the badge e.g. success | fail 
        badge_filename_full (str): The filename of the badge
    
    Returns: 
        None
    """
    badge_message = "pass" if outcome == "success" else "fail"
    badge_color = "brightgreen" if outcome == "success" else "red"
    summary = f"{badge_label}: {badge_message}"

    sitrep = {
        "summary": summary,
        "badge_label": badge_label,
        "tags": tags,
        "digest": digest,
        "outcome": outcome,
    }

    with open("sitrep.json", "w") as f:
        json.dump(sitrep, f)

    badge = {
        "schemaVersion": 1,
        "label": badge_label,
        "message": badge_message,
        "color": badge_color,
    }

    with open(badge_filename_full, "w") as f:
        json.dump(badge, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--badge-label', required=True)
    parser.add_argument('--tags', required=True)
    parser.add_argument('--digest', required=True)
    parser.add_argument('--outcome', required=True)
    parser.add_argument('--badge-filename-full', required=True)
    args = parser.parse_args()
    generate_sitrep(args.badge_label, args.tags, args.digest, args.outcome, args.badge_filename_full)
