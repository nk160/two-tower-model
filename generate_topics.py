import json
from typing import Dict, List
import random

def generate_topic_content(topic: str, num_pairs: int = 10) -> Dict:
    """Generate query-document pairs for a topic using templates and rules"""
    
    # Templates for different query types
    QUERY_TEMPLATES = [
        "what is {concept}",
        "how does {concept} work",
        "explain {concept}",
        "describe {concept} process",
        "compare {concept} and {related_concept}",
        "what are the applications of {concept}",
        "why is {concept} important",
        "what are the principles of {concept}",
        "how to understand {concept}",
        "what are the key features of {concept}"
    ]
    
    # Generate varied content based on topic
    queries = []
    documents = []
    
    # Use topic-specific knowledge to generate pairs
    TOPIC_CONCEPTS = {
        "machine_learning": [
            "neural networks", "deep learning", "supervised learning",
            "reinforcement learning", "feature extraction", "model training",
            "backpropagation", "gradient descent", "overfitting",
            "cross-validation"
        ],
        "quantum_physics": [
            "quantum entanglement", "wave-particle duality", "uncertainty principle",
            "quantum tunneling", "superposition", "quantum field theory",
            "quantum measurement", "quantum states", "quantum coherence",
            "quantum decoherence"
        ],
        "computer_vision": [
            "image processing", "object detection", "facial recognition",
            "image segmentation", "feature extraction", "convolutional networks",
            "edge detection", "pattern recognition", "image classification",
            "visual tracking"
        ],
        "deep_learning": [
            "neural architectures", "backpropagation", "activation functions",
            "gradient descent", "loss functions", "optimization algorithms",
            "regularization", "batch normalization", "transfer learning",
            "model architectures"
        ],
        "artificial_intelligence": [
            "machine learning", "natural language processing", "computer vision",
            "expert systems", "knowledge representation", "reasoning systems",
            "planning algorithms", "robotics", "neural networks", "deep learning"
        ],
        "data_science": [
            "data preprocessing", "feature engineering", "statistical analysis",
            "data visualization", "predictive modeling", "clustering algorithms",
            "dimensionality reduction", "time series analysis", "hypothesis testing",
            "regression analysis"
        ],
        "linear_algebra": [
            "matrices", "vectors", "eigenvalues", "linear transformations",
            "vector spaces", "determinants", "linear systems", "basis vectors",
            "orthogonality", "matrix operations"
        ],
        "statistics": [
            "probability distributions", "hypothesis testing", "regression analysis",
            "correlation", "sampling methods", "confidence intervals", "variance",
            "statistical inference", "experimental design", "statistical significance"
        ],
        "molecular_biology": [
            "DNA replication", "protein synthesis", "gene expression",
            "cellular processes", "molecular pathways", "enzyme kinetics",
            "cell signaling", "genetic regulation", "molecular structures",
            "biochemical reactions"
        ],
        "psychology": [
            "cognitive processes", "behavioral patterns", "mental health",
            "psychological development", "personality theories", "social psychology",
            "learning theories", "memory formation", "emotional intelligence",
            "psychological disorders"
        ],
        "cybersecurity": [
            "network security", "encryption", "threat detection", "vulnerability assessment",
            "penetration testing", "security protocols", "malware analysis", "access control",
            "incident response", "security architecture"
        ],
        "physics": [
            "classical mechanics", "thermodynamics", "electromagnetism", "quantum mechanics",
            "relativity", "wave phenomena", "particle physics", "nuclear physics",
            "fluid dynamics", "statistical mechanics"
        ],
        "economics": [
            "market analysis", "supply demand", "monetary policy", "fiscal policy",
            "economic growth", "international trade", "financial markets", "labor economics",
            "macroeconomic indicators", "economic development"
        ],
        "neuroscience": [
            "neural circuits", "brain structure", "neurotransmitters", "synaptic plasticity",
            "cognitive functions", "neurological disorders", "brain development",
            "neural imaging", "memory formation", "sensory processing"
        ],
        # Add more topics and their concepts
    }
    
    # Generate pairs using templates and concepts
    if topic not in TOPIC_CONCEPTS:
        # Generate topic-specific concepts if not defined
        default_concepts = [
            f"{topic} fundamentals",
            f"{topic} principles",
            f"{topic} methods",
            f"{topic} applications",
            f"{topic} theory",
            f"{topic} practice",
            f"{topic} analysis",
            f"{topic} techniques",
            f"{topic} development",
            f"{topic} research"
        ]
        concepts = default_concepts
    else:
        concepts = TOPIC_CONCEPTS[topic]
    
    for i in range(num_pairs):
        concept = concepts[i % len(concepts)]
        related_concept = concepts[(i+1) % len(concepts)]
        
        query = QUERY_TEMPLATES[i % len(QUERY_TEMPLATES)].format(
            concept=concept,
            related_concept=related_concept
        )
        
        # Generate detailed document response
        document = generate_document(concept, topic, related_concept)
        
        queries.append(query)
        documents.append(document)
    
    return {
        "queries": queries,
        "documents": documents
    }

def generate_document(concept, topic, related_concept):
    """Generate more varied and detailed documents"""
    templates = [
        # Detailed explanation
        f"""
        {concept} is a key concept in {topic} that encompasses multiple aspects. 
        At its core, it involves {concept.split('_')[0]} principles applied to {topic.split('_')[0]} problems. 
        The process typically includes several steps: analysis, implementation, and validation.
        Recent developments have shown connections with {related_concept}, particularly in advanced applications.
        """,
        
        # Problem-solution format
        f"""
        When working with {concept} in {topic}, practitioners often face several challenges.
        Common issues include complexity management and optimization requirements.
        Modern approaches combine traditional methods with new techniques from {related_concept}.
        Best practices emphasize systematic testing and continuous improvement.
        """,
        
        # Compare and contrast
        f"""
        While {concept} shares some similarities with {related_concept}, they serve different purposes in {topic}.
        The main distinction lies in their application: {concept} focuses on core functionality,
        whereas {related_concept} addresses higher-level concerns. Understanding both is crucial for mastery of {topic}.
        """
    ]
    return random.choice(templates).strip()

def generate_all_topics() -> Dict:
    """Generate complete topics dictionary with 100+ topics"""
    
    # Define topic categories
    CATEGORIES = {
        "STEM_Computing": [
            "machine_learning", "deep_learning", "computer_vision",
            "natural_language_processing", "robotics", "artificial_intelligence",
            "cloud_computing", "cybersecurity", "blockchain_technology",
            "quantum_computing", "edge_computing", "internet_of_things",
            "big_data_analytics", "software_engineering", "database_systems",
            "computer_networks", "operating_systems", "web_development",
            "mobile_computing", "distributed_systems"
        ],
        
        "STEM_Mathematics": [
            "linear_algebra", "calculus", "statistics", "probability_theory",
            "number_theory", "discrete_mathematics", "optimization",
            "graph_theory", "topology", "differential_equations",
            "numerical_analysis", "abstract_algebra", "mathematical_logic",
            "game_theory", "cryptography"
        ],
        
        "Physical_Sciences": [
            "quantum_physics", "particle_physics", "astrophysics",
            "thermodynamics", "electromagnetism", "optics", "mechanics",
            "relativity_theory", "nuclear_physics", "condensed_matter",
            "fluid_dynamics", "plasma_physics", "string_theory",
            "cosmology", "materials_science"
        ],
        
        "Life_Sciences": [
            "molecular_biology", "genetics", "biochemistry", "cell_biology",
            "neuroscience", "immunology", "microbiology", "ecology",
            "evolutionary_biology", "physiology", "bioinformatics",
            "developmental_biology", "marine_biology", "botany",
            "zoology"
        ],
        
        "Earth_Sciences": [
            "geology", "meteorology", "oceanography", "climatology",
            "environmental_science", "atmospheric_science", "hydrology",
            "seismology", "volcanology", "paleontology"
        ],
        
        "Social_Sciences": [
            "psychology", "sociology", "economics", "political_science",
            "anthropology", "archaeology", "linguistics", "human_geography",
            "cognitive_science", "behavioral_economics", "criminology",
            "international_relations", "urban_planning", "social_psychology",
            "developmental_psychology"
        ],
        
        "Business_Economics": [
            "macroeconomics", "microeconomics", "finance", "marketing",
            "management", "entrepreneurship", "accounting", "investment",
            "business_strategy", "operations_management", "supply_chain",
            "human_resources", "project_management", "risk_management",
            "digital_marketing"
        ],
        
        "Arts_Humanities": [
            "art_history", "literature", "philosophy", "music_theory",
            "film_studies", "theater", "architecture", "cultural_studies",
            "religious_studies", "classical_studies", "modern_art",
            "contemporary_art", "creative_writing", "poetry",
            "media_studies"
        ],
        
        "History_Civilization": [
            "ancient_history", "medieval_history", "modern_history",
            "world_war_history", "renaissance_period", "industrial_revolution",
            "cold_war", "ancient_civilizations", "european_history",
            "asian_history", "american_history", "african_history"
        ],
        
        "Health_Medicine": [
            "anatomy", "pharmacology", "pathology", "epidemiology",
            "public_health", "nutrition", "mental_health", "surgery",
            "pediatrics", "cardiology", "neurology", "immunology",
            "oncology", "emergency_medicine", "preventive_medicine"
        ]
    }
    
    topics = {}
    for category, topic_list in CATEGORIES.items():
        for topic in topic_list:
            topics[topic] = generate_topic_content(topic)
    
    return topics

if __name__ == "__main__":
    # Generate all topics
    all_topics = generate_all_topics()
    
    # Save to file
    with open("topics.json", "w") as f:
        json.dump(all_topics, f, indent=2) 