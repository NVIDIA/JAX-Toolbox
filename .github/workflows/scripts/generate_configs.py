import json
import os


def main():
    """ Main function for image tagging"""

    all_tags = json.loads(os.environ['ALL_TAGS'])
    publish_configs = []

    flavors = [
        'base', 'jax', 'triton', 'equinox', 'maxtext', 'levanter',
        'upstream-t5x', 'upstream-pax', 't5x', 'pax', 'gemma'
    ]
    stages = ['mealkit', 'final']

    for stage in stages:
        for flavor in flavors:
            matching_tags = [
                tag for tag in all_tags if
                tag['stage'] == stage and tag['flavor'] == flavor and tag['tag']
            ]
            if matching_tags:
                source_image = [tag['tag'] for tag in matching_tags]
                priority = max(tag['priority'] for tag in matching_tags)
                target_image = os.environ.get('MEALKIT_IMAGE_REPO' if stage == 'mealkit' else 'FINAL_IMAGE_REPO')
                publish_configs.append({
                    'flavor': flavor,
                    'target_image': target_image,
                    'priority': priority,
                    'source_image': source_image,
                    'stage': stage
                })

    print(json.dumps({'config': publish_configs}))


if __name__ == '__main__':
    main()
