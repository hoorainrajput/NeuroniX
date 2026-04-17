"""
train_model.py
--------------
Trains the Neural Pattern Classifier for the Neuromorphic AI Learning Platform.
Classifies text into 4 neural signal categories:
  EXCITATORY / INHIBITORY / OSCILLATORY / SPARSE

Run this ONCE before starting Flask:
    python model/train_model.py
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os


TRAINING_DATA = [
    # ── EXCITATORY 
    ("the neuron fires rapidly sending strong signals through the network", "EXCITATORY"),
    ("action potentials cascade through synaptic pathways at maximum frequency", "EXCITATORY"),
    ("intense burst of activity excites surrounding neurons dramatically", "EXCITATORY"),
    ("rapid signal propagation accelerates across neural circuits powerfully", "EXCITATORY"),
    ("strong positive feedback amplifies neural firing dramatically and fast", "EXCITATORY"),
    ("high frequency oscillations drive massive network activation", "EXCITATORY"),
    ("energetic neural burst triggers downstream excitation powerfully", "EXCITATORY"),
    ("explosive synaptic release generates strong depolarization", "EXCITATORY"),
    ("maximum voltage threshold reached triggering full action potential", "EXCITATORY"),
    ("cascade of excitatory neurotransmitters floods the synapse actively", "EXCITATORY"),
    ("neurons fire together wire together strengthening pathways rapidly", "EXCITATORY"),
    ("spike train intensifies as stimulus grows stronger and louder", "EXCITATORY"),
    ("glutamate binding opens ion channels flooding the cell excitingly", "EXCITATORY"),
    ("long term potentiation strengthens synapses with repeated firing", "EXCITATORY"),
    ("neural avalanche spreads through cortex in rapid excitation burst", "EXCITATORY"),
    ("the stimulus triggers a powerful wave of depolarization", "EXCITATORY"),
    ("excitatory postsynaptic potential drives the membrane toward threshold", "EXCITATORY"),
    ("high-intensity input floods downstream circuits with activation", "EXCITATORY"),
    ("voltage-gated sodium channels open in a rapid cascade", "EXCITATORY"),
    ("strong synaptic drive produces sustained high-frequency firing", "EXCITATORY"),
    ("bursting discharge propagates through the thalamocortical loop", "EXCITATORY"),
    ("the network reaches a critical point triggering mass activation", "EXCITATORY"),
    ("dense spike trains encode the intensity of a strong stimulus", "EXCITATORY"),
    ("acetylcholine release amplifies excitatory tone across the cortex", "EXCITATORY"),
    ("population burst fires synchronously across hundreds of cells", "EXCITATORY"),

    # ── INHIBITORY ────────────────────────────────────────────────────────────
    ("the inhibitory signal suppresses neural firing significantly", "INHIBITORY"),
    ("GABA release silences surrounding neurons reducing activity", "INHIBITORY"),
    ("hyperpolarization prevents the neuron from reaching threshold", "INHIBITORY"),
    ("slow inhibitory postsynaptic potential dampens excitation", "INHIBITORY"),
    ("lateral inhibition reduces noise in sensory processing pathways", "INHIBITORY"),
    ("interneurons suppress excessive firing to prevent seizures", "INHIBITORY"),
    ("neural silence spreads as inhibition dominates the network", "INHIBITORY"),
    ("weak signal fades below threshold causing no response", "INHIBITORY"),
    ("potassium channels open driving voltage below resting potential", "INHIBITORY"),
    ("feedback inhibition reduces runaway excitation in cortical loops", "INHIBITORY"),
    ("quiet period follows the burst as refractory phase begins", "INHIBITORY"),
    ("long term depression weakens synaptic connections over time", "INHIBITORY"),
    ("glycine inhibits spinal motor neurons preventing movement", "INHIBITORY"),
    ("surround inhibition sharpens contrast in visual processing areas", "INHIBITORY"),
    ("chloride influx drives the membrane potential more negative", "INHIBITORY"),
    ("the neuron remains below threshold despite incoming input", "INHIBITORY"),
    ("refractory period silences the cell after a spike event", "INHIBITORY"),
    ("inhibitory interneuron reduces activity in the surrounding circuit", "INHIBITORY"),
    ("tonic inhibition keeps baseline firing rates extremely low", "INHIBITORY"),
    ("shunting inhibition reduces the amplitude of excitatory inputs", "INHIBITORY"),
    ("down-regulation of AMPA receptors decreases synaptic strength", "INHIBITORY"),
    ("the cortex enters a down state with suppressed neural activity", "INHIBITORY"),
    ("inhibitory drive prevents pathological hyperexcitability in the loop", "INHIBITORY"),
    ("slow GABA-B currents produce prolonged suppression of firing", "INHIBITORY"),
    ("the circuit falls silent as inhibition outpaces excitation", "INHIBITORY"),

    # ── OSCILLATORY ───────────────────────────────────────────────────────────
    ("theta rhythm oscillates between hippocampus and prefrontal cortex", "OSCILLATORY"),
    ("gamma oscillations synchronize neural assemblies during perception", "OSCILLATORY"),
    ("alpha waves cycle rhythmically at ten hertz during rest", "OSCILLATORY"),
    ("periodic bursting pattern repeats in regular rhythmic cycles", "OSCILLATORY"),
    ("circadian rhythm cycles every twenty four hours controlling sleep", "OSCILLATORY"),
    ("beta oscillations repeat rhythmically during motor planning tasks", "OSCILLATORY"),
    ("delta waves oscillate slowly during deep sleep stages cycles", "OSCILLATORY"),
    ("neural synchrony oscillates in phase locking patterns rhythmically", "OSCILLATORY"),
    ("cortical oscillations coordinate timing across distant brain regions", "OSCILLATORY"),
    ("spindle waves repeat in cycles during memory consolidation sleep", "OSCILLATORY"),
    ("respiratory rhythm oscillates and drives hippocampal theta waves", "OSCILLATORY"),
    ("pacemaker neurons set the beat for central pattern generators", "OSCILLATORY"),
    ("theta burst stimulation mimics natural oscillatory learning patterns", "OSCILLATORY"),
    ("cross-frequency coupling links slow and fast neural oscillations", "OSCILLATORY"),
    ("basal ganglia oscillations cycle between movement and rest phases", "OSCILLATORY"),
    ("the network oscillates between up and down states rhythmically", "OSCILLATORY"),
    ("periodic inhibition gates incoming signals in a rhythmic pattern", "OSCILLATORY"),
    ("mu rhythm oscillates over the motor cortex during observation", "OSCILLATORY"),
    ("slow cortical waves travel in repeating cycles across hemispheres", "OSCILLATORY"),
    ("ripple oscillations repeat rapidly during hippocampal replay events", "OSCILLATORY"),
    ("the thalamic pacemaker drives rhythmic cortical oscillation patterns", "OSCILLATORY"),
    ("neural phase precession advances each cycle through the field", "OSCILLATORY"),
    ("gamma bursts recur periodically synchronizing cell assemblies together", "OSCILLATORY"),
    ("traveling waves sweep cortex in rhythmic periodic fashion", "OSCILLATORY"),
    ("the circuit locks into a repeating limit cycle attractor state", "OSCILLATORY"),

    # ── SPARSE ────────────────────────────────────────────────────────────────
    ("one neuron fires while others remain silent and inactive", "SPARSE"),
    ("minimal activation pattern encodes information efficiently and simply", "SPARSE"),
    ("few synapses respond to the input signal selectively", "SPARSE"),
    ("sparse representation uses only a small fraction of neurons", "SPARSE"),
    ("single spike encodes the presence of a specific feature", "SPARSE"),
    ("place cells fire only in one specific location in space", "SPARSE"),
    ("grandmother cell hypothesis suggests single neurons encode concepts", "SPARSE"),
    ("minimal energy neural code activates just a handful of cells", "SPARSE"),
    ("low baseline firing rate characterizes sparse cortical coding", "SPARSE"),
    ("selective attention filters out most inputs keeping only key signal", "SPARSE"),
    ("a small subset of neurons carries the essential information", "SPARSE"),
    ("two neurons encode the full scene with high efficiency", "SPARSE"),
    ("simple input single response elegant neural computation achieved", "SPARSE"),
    ("quiet network few cells active encoding is maximally efficient", "SPARSE"),
    ("minimal redundancy each neuron carries unique independent information", "SPARSE"),
    ("only one or two cells respond to the presented stimulus", "SPARSE"),
    ("the population vector is almost entirely zero except one unit", "SPARSE"),
    ("a single grandmother cell fires exclusively for one face", "SPARSE"),
    ("grid cell activates at a specific spatial location only", "SPARSE"),
    ("the representation is distributed across very few active units", "SPARSE"),
    ("concept cell fires to a single person regardless of format", "SPARSE"),
    ("independent component analysis reveals sparse basis functions", "SPARSE"),
    ("low activity fraction keeps metabolic cost near the minimum", "SPARSE"),
    ("one cell acts as a detector for exactly one stimulus feature", "SPARSE"),
    ("the network relies on a minimal population code for efficiency", "SPARSE"),
]

# ── Prepare Dataset ───────────────────────────────────────────────────────────

texts  = [d[0] for d in TRAINING_DATA]
labels = [d[1] for d in TRAINING_DATA]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ── Build Pipeline ────────────────────────────────────────────────────────────

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=1000,
        sublinear_tf=True,
        min_df=1,
    )),
    ("clf", LogisticRegression(
        C=3.0,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )),
])

# ── Train 

print("Training Neural Pattern Classifier...")
pipeline.fit(X_train, y_train)

# ── Evaluate 

y_pred = pipeline.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cv     = cross_val_score(pipeline, texts, labels, cv=5, scoring="accuracy")

print(f"\n  Test Accuracy  : {acc * 100:.1f}%")
print(f"  CV Mean±Std    : {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Save Model 

save_dir   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(save_dir, "neural_classifier.joblib")
joblib.dump(pipeline, model_path)
print(f"Model saved → {model_path}")

# ── Class Metadata 

CLASS_INFO = {
    "EXCITATORY": {
        "emoji": "⚡",
        "color": "#FF6B35",
        "description": "High-energy signal pattern. Neurons fire rapidly, cascading excitation through the network.",
        "analogy": "Like a crowd cheering that grows louder — each neuron amplifies the signal forward.",
    },
    "INHIBITORY": {
        "emoji": "🛑",
        "color": "#4ECDC4",
        "description": "Suppressive signal pattern. Neural firing is dampened or silenced to prevent runaway excitation.",
        "analogy": "Like a volume knob turned dfvown — keeps the brain from overloading.",
    },
    "OSCILLATORY": {
        "emoji": "〰️",
        "color": "#A78BFA",
        "description": "Rhythmic, cyclic signal pattern. Neurons fire in synchronized waves at regular intervals.",
        "analogy": "Like a heartbeat — steady rhythm coordinates distant brain regions.",
    },
    "SPARSE": {
        "emoji": "✦",
        "color": "#FCD34D",
        "description": "Minimal activation pattern. Only a handful of neurons encode the information efficiently.",
        "analogy": "Like morse code — a few precise signals carry maximum meaning.",
    },
}

meta_path = os.path.join(save_dir, "class_info.joblib")
joblib.dump(CLASS_INFO, meta_path)
print(f"Metadata saved → {meta_path}")
print("\nDone! You can now run: python app.py")

