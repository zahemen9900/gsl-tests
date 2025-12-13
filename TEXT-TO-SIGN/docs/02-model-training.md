# Text-to-Sign: GAN-NAT Training Architecture

## 1. Architectural Philosophy: The "Anti-Zombie" Approach

The fundamental challenge in Sign Language Generation (SLG) is the "Regression to the Mean" (Zombie Effect). When trained with simple MSE loss, models output the *average* of all possible movements, resulting in stiff, lifeless, and undefined handshapes.

We address this with a **GAN-NAT (Generative Adversarial Non-Autoregressive Transformer)** architecture.
- **GAN**: Forces high-frequency realism (sharp movements).
- **NAT**: Parallel generation for high-speed inference (no autoregressive bottleneck).

---

## 2. Generator Network ($G$)
The Generator maps conditional inputs (Text, Style) to a sequence of 3D poses.

### 2.1 Inputs
1.  **Text conditioning**: $E_{text} \in \mathbb{R}^{L \times 256}$ (from DistilBERT Adapter).
2.  **Style Seed ($z$)**: $z \sim \mathcal{N}(0, I) \in \mathbb{R}^{64}$. A random noise vector sampled once per sentence.

### 2.2 Components
#### A. Length Predictor
A small classification head that predicts the duration $T$ of the sign sequence.
- *Input*: Pooled Text Embedding ($E_{text, pooled}$).
- *Output*: Softmax distribution over lengths $[10, \dots, 200]$.
- *Training*: Cross-Entropy against ground truth length.

#### B. Positional Encoding & Seeding
- Create a specific interaction between the text and the latent time queries.
- *Query Initialization*: `LearnedTimeEmbeddings(T)` + `StyleProj(z)`.
- This ensures the entire sequence "knows" the intended style (e.g., angry/fast) from the start.

#### C. Non-Autoregressive Decoder
- **Layers**: 4-6 Transformer Decoder layers.
- **Self-Attention**: Full visibility (bi-directional), not masked.
- **Cross-Attention**: Attends to $E_{text}$.
- **Output Head**: Projects hidden states (256) to Output Poses (540).

---

## 3. Discriminator Network ($D$)
The Critic's job is to look at a sequence of poses and classify it as "Real Human Motion" or "Fake Machine Motion".

### 3.1 Architecture: 1D-CNN (Temporal ConvNet)
We use a temporal convolution network rather than a Transformer for the discriminator to focus on *local motion dynamics* (velocity/acceleration consistency).

- **Input**: $(B, T, 540)$
- **Layers**:
    - `Conv1D(k=3, s=1)` $\rightarrow$ `LeakyReLU` $\rightarrow$ `SpectralNorm`
    - `Conv1D(k=3, s=2)` (Downsample) $\dots$
- **Output**: Multi-scale validity scores.

---

## 4. Loss Functions
The total loss is a weighted sum of four components:

$$ \mathcal{L}_{total} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{adv}\mathcal{L}_{adv} + \lambda_{geo}\mathcal{L}_{geo} + \lambda_{kl}\mathcal{L}_{kl} $$

### 4.1 Reconstruction Loss ($\mathcal{L}_{rec}$)
Huber Loss (Smooth L1) on:
1.  **Positions**: $P_{pred} \approx P_{gt}$
2.  **Velocities** (1st derivative): $\Delta P_{pred} \approx \Delta P_{gt}$
3.  **Accelerations** (2nd derivative): $\Delta^2 P_{pred} \approx \Delta^2 P_{gt}$

*Why Huber?* It is less sensitive to outliers than MSE, preventing the model from over-correcting for noisy tracking data.

### 4.2 Adversarial Loss ($\mathcal{L}_{adv}$)
Hinge GAN Loss:
- **Generator**: Minimize $-D(G(z))$ (Try to fool $D$).
- **Discriminator**: Maximize $D(x) - D(G(z))$.

### 4.3 Geometric Losses ($\mathcal{L}_{geo}$)
Physics-based constraints to prevent anatomical violations.
1.  **Bone Length Consistency**: $\sum_{b \in Bones} | \|J_{parent} - J_{child}\| - L_{ref_b} |$.
    - Penalizes stretching/shrinking limbs.
2.  **Foot Sliding (Contact)**:
    - If $y_{heel} < \epsilon$ (touching floor), penalize $v_{heel}^2$.

### 4.4 Style Consistency (Optional)
If using style labels (e.g., "Angry", "Happy"), add a classification loss on the generated motion.

---

## 5. Training Curriculum
To prevent mode collapse, we train in phases:

### Phase 1: Warmup (0-15k Steps)
- **Goal**: Learn the mean pose and rough alignment.
- **Active Losses**: $\mathcal{L}_{rec}$ (Reconstruction) ONLY.
- **Discriminator**: Frozen/Off.
- *Result*: "Zombie" motion (smooth, average, blurry).

### Phase 2: Geometric Stabilization (15k-30k Steps)
- **Goal**: Fix bone stretching and floaty feet.
- **Active Losses**: $\mathcal{L}_{rec} + \mathcal{L}_{geo}$.
- **Discriminator**: Frozen/Off.

### Phase 3: Adversarial Sharpening (30k+ Steps)
- **Goal**: Add texture, micro-movements, and realism.
- **Active Losses**: All ($\mathcal{L}_{rec} + \mathcal{L}_{geo} + \mathcal{L}_{adv}$).
- **Discriminator**: Active.
- *Note*: Reduce $\lambda_{rec}$ slightly to allow the GAN to add detail that might differ slightly from MSE but looks more real.

---

## 6. Implementation Notes for `ghsl_gan_trainer.py`
- **Optimizer**: AdamW ($\beta_1=0.5, \beta_2=0.9$) for GAN stability.
- **Learning Rate**: Two separate schedulers. $G$ usually needs higher LR than $D$ (Two-Time-Scale Update Rule - TTUR).
- **Gradient Penalty**: If training is unstable, implement R1 Gradient Penalty on the Discriminator.
