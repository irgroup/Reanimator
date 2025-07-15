
import random
import json
from pathlib import Path
import pickle
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# Try to import REANIMATOR's Judgement class for type safety
from reanimator.models import *

## Human Annotation

def get_pairs_for_modality(machine_judgements, chunks, modality):
    """
    Returns the machine judgements for chosen modality.
    """
    # Only include pairs of the selected modality
    return [j for j in machine_judgements if any(c['chunk_id'] == j['chunk_id'] and c['modality'] == modality for c in chunks)]

def load_existing_labels(path):
    """
    loads (human) relevance labels
    """
    if Path(path).exists():
        with open(path, 'r') as f:
            return json.load(f)
    return []

def get_labeled_chunks(judgements):
    return [j['chunk_id'] for j in judgements]


def load_label_pairs(machine_judgements, chunks, output_path, modality, num_pairs):

    # Load already-labeled pairs
    try: 
        existing_labels = load_existing_labels(output_path)
    except:
        existing_labels = []
    labeled_chunks = get_labeled_chunks(existing_labels)

    # Filter and sample
    pairs = get_pairs_for_modality(machine_judgements, chunks, modality)
    unlabeled_pairs = [j for j in pairs if j["chunk_id"] not in labeled_chunks]

    num_to_label = min(num_pairs, len(pairs))  # Change as needed
    sampled = random.sample(pairs, num_to_label) if unlabeled_pairs else []
    to_label = [a for a in sampled if not (a["chunk_id"] in labeled_chunks)] 

    print(f"{len(sampled)-len(existing_labels)}  texts left to label: {len(sampled)} pairs to label, {len(existing_labels)} already labeled in '{output_path}'.")
    return to_label

def get_topic(query_id, topics):
    for t in topics:
        if t['query_id'] == query_id:
            return t
    return {}

def get_chunk(doc_id, chunks):
    for c in chunks:
        if c['chunk_id'] == doc_id:
            return c
    return {}

def render_topic(topic):
    html = f"<b>Query ID:</b> {topic.get('query_id', '')}<br>"
    html += f"<b>Query Text:</b> {topic.get('query_text', '')}<br>"
    context = topic.get('context', {})
    if context:
        html += f"<b>Description:</b> {context.get('description', '')}<br>"
        html += f"<b>Narrative:</b> {context.get('narrative', '')}<br>"
    if topic.get('rewritten_texts'):
        html += f"<b>Rewritten Texts:</b> {topic['rewritten_texts']}<br>"
    if topic.get('metadata'):
        html += f"<b>Metadata:</b> {topic['metadata']}<br>"
    return html

def render_chunk(chunk):
    if chunk.get('modality') == 'table':
        # Render as HTML table if possible
        return chunk.get('text', '')  # Should be HTML already in example
    else:
        return f"<pre>{chunk.get('text', '')}</pre>"

def save_labels(labels, path):
    with open(path, 'w') as f:
        json.dump(labels, f, indent=4)

def labeling_interface(output_path, sampled, topics, chunks):
    human_judgements = load_existing_labels(output_path)

    for pair in sampled:
        topic = get_topic(pair['query_id'], topics)
        chunk = get_chunk(pair['chunk_id'], chunks)
        
        
        label_widget = widgets.RadioButtons(
            options=[
                ('0: Not relevant', 0),
                ('1: Partially relevant', 1),
                ('2: Fully relevant', 2)
            ],
            description='Relevance:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%'),
            disabled=False
        )
        submit_button = widgets.Button(description='Submit', button_style='primary')
        output = widgets.Output()
        
        display(HTML('<hr>'))
        display(HTML('<b>TOPIC</b>'))
        display(HTML(render_topic(topic)))
        display(HTML(f'<b>{chunk.get("modality", "Chunk").capitalize()}:</b>'))
        display(HTML(render_chunk(chunk)))
        display(label_widget)
        display(submit_button)
        display(output)
        
        def on_submit(b, pair=pair, label_widget=label_widget, submit_button=submit_button, output=output):
            with output:
                clear_output()
                print('Label saved!')
            human_judgements.append({
                'query_id': pair['query_id'],
                'chunk_id': pair['chunk_id'],
                'doc_id': pair['doc_id'],
                'score': label_widget.value,
                'source': 'human'
            })
            save_labels(human_judgements, output_path)
            submit_button.disabled = True
            label_widget.disabled = True
        submit_button.on_click(on_submit)
