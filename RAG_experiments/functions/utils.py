import xml.etree.ElementTree as ET
from typing import Iterable
import re



def get_topic_info(topics_xml, topic_number):
    """
    Extract the query, question, and narrative for a given topic number from the XML string.
    """
    # Parse the XML
    root = ET.fromstring(topics_xml)
    
    # Find the specific topic using XPath
    topic = root.find(f"./topic[@number='{topic_number}']")
    if topic is None:
        raise ValueError(f"Topic number {topic_number} not found.")
    
    # Extract query, question, and narrative
    query = topic.findtext("query", default="No query found").strip()
    question = topic.findtext("question", default="No question found").strip()
    narrative = topic.findtext("narrative", default="No narrative found").strip()
    
    return query, question, narrative