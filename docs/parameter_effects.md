# Parameter Effects on LLM Fiction Generation

This document details empirical observations and comparative analyses of key LLM hyperparameters—**Temperature**, **Top P** (Nucleus Sampling), and **Repeat Penalty** (Frequency/Repetition Penalty)—when co-authoring creative fiction using open-source models like Llama 3.1 and Mistral.

---

### Temperature

Temperature controls the randomness of token selection during sampling. A lower value sharpens the probability distribution, making the model pick more likely, deterministic tokens. A higher value flattens the distribution, giving less probable tokens a higher chance of selection, which increases stylistic variety and vocabulary richness at the cost of potential hallucinations.

#### Base Prompt
> *"The clocktower struck midnight as the detective unlocked the archives door."*

#### Low Temperature (Temperature = 0.2)
```text
The clocktower struck midnight as the detective unlocked the archives door. Cold rain drummed steadily against the stained glass windows above. Detective Vance pulled his collar up against the draft, stepping quietly into the dark corridor. Rows of metal filing cabinets stretched into the shadows, each containing decades of forgotten city records. He clicked on his brass penlight and began searching for the 1924 ledger.
```
*Analysis*: Highly coherent, strictly follows logical narrative progression, uses standard genre phrasing, minimal risk of structural degradation, but slightly predictable.

#### High Temperature (Temperature = 1.3)
```text
The clocktower struck midnight as the detective unlocked the archives door. Iron teeth ground within the lock like ancient molars chewing through centuries of silence. Vance didn't breathe; the dust smelled of oxidized treason and lavender ink. Shadows slithered along the vaulted rafters, whispering of the phantom council that had governed the city before the Great Flood of embers.
```
*Analysis*: Vividly poetic and metaphorically daring ("oxidized treason", "ancient molars"). Shows elevated stylistic flair, ideal for high fantasy or surreal noir, but risks narrative drift if pushed beyond 1.5.

---

### Top P

Top P (Nucleus Sampling) restricts the candidate token pool to the smallest set of tokens whose cumulative probability exceeds the threshold `p`. Unlike temperature (which alters all probabilities), Top P truncates the tail of unlikely words dynamically based on context confidence.

#### Base Prompt
> *"Describe the magical artifact resting upon the altar of starlight."*

#### Low Top P (Top P = 0.3)
```text
The artifact was an orb of polished crystal that rested upon the stone altar. It pulsed with a steady, pale blue light. Within its core, tiny points of light drifted slowly, resembling distant stars in the night sky. The air around the altar was cold and still.
```
*Analysis*: Focused, concise, and clean. Avoids esoteric vocabulary and sticks to the highest confidence tokens. Excellent for crisp, objective descriptions.

#### High Top P (Top P = 0.95)
```text
Cradled in obsidian talons atop the altar rested the Astrolabe of the First Dawn. Its concentric rings of iridescent starmetal spun without friction, emitting an ethereal resonance that vibrated in the marrow of anyone who stood within its celestial aura. Microscopic constellations flared and died along its gyroscopic perimeter.
```
*Analysis*: Vastly expanded vocabulary range ("concentric rings of iridescent starmetal", "gyroscopic perimeter"). Enables lush, sensory worldbuilding while maintaining grammatical consistency.

---

### Repeat Penalty

Repeat Penalty penalizes tokens that have already appeared in the recent context or generation window. This prevents the LLM from entering repetitive loops, monotonous sentence structures, or overusing character names and crutch phrases.

#### Base Prompt
> *"Describe the duel between the disgraced knight and the shadow assassin in the ruined cathedral."*

#### Low Repeat Penalty (Repeat Penalty = 1.0 - No Penalty)
```text
The knight swung his sword at the assassin. The assassin dodged the sword and struck back with his blade. The knight blocked the blade with his shield and swung his sword again. In the ruined cathedral, the knight and the assassin fought fiercely. The knight felt his strength fading as the assassin moved faster and faster around the broken pillars of the cathedral.
```
*Analysis*: Suffers from repetitive sentence syntax ("The knight...", "The assassin..."), recurrent noun phrases ("sword", "blade", "cathedral"), and mechanical combat choreography.

#### High Repeat Penalty (Repeat Penalty = 1.25)
```text
Sir Roland cleaved downward, his broadsword shattering stone where the cloaked specter had stood an instant prior. Sidestepping the flurry, the assassin countered with twin curved daggers aimed at exposed chainmail seams. Roland pivoted, absorbing the deflection upon his heraldic buckler before driving a gauntleted elbow into his adversary's ribs amidst the shattered sanctuary.
```
*Analysis*: Dynamic syntactic variation, rich lexical substitution (broadsword -> blade -> steel; assassin -> cloaked specter -> adversary), eliminating repetitive pacing and producing professional-grade prose.

---

### Recommended Presets for Fiction Writing

| Fiction Sub-Genre / Purpose | Temperature | Top P | Repeat Penalty | Best Suited For |
| :--- | :---: | :---: | :---: | :--- |
| **Grounded Mystery / Hard Sci-Fi** | `0.4 - 0.6` | `0.75` | `1.15` | Consistent logic, technical accuracy, tight clues. |
| **High Fantasy / Epic Worldbuilding** | `0.7 - 0.85` | `0.90` | `1.18` | Vivid prose, mythic lore injection, sensory depth. |
| **Fast-Paced Action & Snappy Dialogue** | `0.65` | `0.85` | `1.20` | Crisp exchanges, varied verb choices, rhythmic flow. |
| **Surreal / Dream Sequences / Mythic Prose** | `1.1 - 1.3` | `0.95` | `1.12` | Unconventional metaphors, poetic imagery. |
