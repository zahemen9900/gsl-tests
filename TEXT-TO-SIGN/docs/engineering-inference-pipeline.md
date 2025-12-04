# Engineering Plan: Rigging & Real-Time IK for GhSL

Goal: Map 540-point ML output to a Standard Unity/Web Avatar (Mixamo/VRM).

Constraint: Real-time performance (60 FPS) on Web (Three.js/React-Fiber) or Unity.

## 1. The Core Problem: Points vs. Rotations

- **ML Output:** 3D Coordinates $(x, y, z)$ in a normalized space (MediaPipe Topology).
    
- **Avatar Input:** Bone Rotations (Quaternions) for a hierarchical skeleton.
    

**We will use a "Hybrid IK" approach:**

1. **Torso/Head:** FK (Forward Kinematics) based on look-at vectors.
    
2. **Arms/Legs:** Two-Bone IK (Inverse Kinematics).
    
3. **Fingers:** FK based on local angles.
    

## 2. The Pipeline Steps

### Step 1: The "T-Pose" Calibration (Initialization)

Before any animation plays, we must align the ML model's coordinate space with the Avatar's space.

1. **Load Avatar:** Ensure it is in T-Pose.
    
2. **Measure Bone Lengths:** Calculate the static length of every bone (e.g., `Length_UpperArm = dist(Shoulder, Elbow)`).
    
3. **Scale Factor:**
    
    - ML models usually output normalized units (height $\approx 1.0$).
        
    - Avatar units might be meters (height $\approx 1.7$).
        
    - Calculate `GlobalScale = AvatarHeight / ModelAvgHeight`.
        
    - Apply this scale to all ML outputs at runtime.
        

### Step 2: Inverse Kinematics (The Arms)

This is critical for Sign Language. The hands must reach specific targets relative to the body.

Algorithm: Two-Bone IK (Analytical)

We don't need expensive iterative solvers (CCD/FABRIK) for just arms. We can use high-speed trigonometry.

**Inputs:**

- `Root`: Avatar Shoulder position.
    
- `Target`: ML-predicted Hand position.
    
- `Hint`: ML-predicted Elbow position (The "Pole Vector").
    

**Logic:**

1. **Triangle Rule:** We know the lengths of the Upper Arm ($a$) and Lower Arm ($b$), and the distance to the Target ($c$). Using the Law of Cosines, we calculate the interior angle of the elbow.
    
2. **Pole Vector:** The elbow could bend at that angle in _any_ direction (infinite circle). We use the ML-predicted Elbow position to "pin" the rotation plane.
    
3. **Apply Rotation:** Rotate Upper Arm to point to Elbow; Rotate Lower Arm to point to Hand.
    

### Step 3: Finger Solving (The "Local" FK)

IK is too heavy for 10 fingers (30 joints). We use **Vector Alignment**.

- **ML Data:** We have 3 points for the index finger: `[Knuckle, Middle, Tip]`.
    
- **Calculation:**
    
    1. Create a vector $V_{bone} = P_{middle} - P_{knuckle}$.
        
    2. Calculate the rotation needed to turn the Avatar's rest bone vector to align with $V_{bone}$.
        
    3. Apply this rotation (clamped to natural human limits).
        

## 3. Smoothing (The "Jitter Killer")

Raw ML output vibrates.

**Do NOT use:**

- Simple Moving Average (Makes motion floaty/laggy).
    
- Gaussian Smoothing (Adds too much latency).
    

DO use: One Euro Filter (1€ Filter)

This is a first-order low-pass filter with an adaptive cutoff frequency.

- **High Speed:** Cutoff increases $\rightarrow$ Low latency (crisp fast signs).
    
- **Low Speed:** Cutoff decreases $\rightarrow$ High smoothing (steady hold signs).
    

**Code Snippet (Concept):**

```
import { OneEuroFilter } from 'three-stdlib';

const filterX = new OneEuroFilter(60, 1.0, 0.001, 1.0); // minCutoff, beta

function updateFrame(rawPos) {
    const smoothX = filterX.filter(rawPos.x, timestamp);
    // ... apply to Y and Z
    return new Vector3(smoothX, smoothY, smoothZ);
}
```

## 4. Platform Implementation Details

### For Web (React Three Fiber / Three.js)

Use the library **`Kalidokit`** or **`Mediapipe-Holistic`** utils.

- _Why:_ They have built-in "Solver" functions that take raw MediaPipe points and output rig-ready rotations.
    
- _Workflow:_
    
    1. ML Model (Python/ONNX) $\rightarrow$ sends JSON array of points `[33, 3]` per frame.
        
    2. Frontend Loop:
        
        ```
        // 1. Smooth Points
        const smoothedLandmarks = oneEuroFilter(prediction);
        
        // 2. Solve Rig
        const rig = Kalidokit.Pose.solve(smoothedLandmarks, {
            runtime: "mediapipe",
            video: {height: 1080, width: 1920}
        });
        
        // 3. Apply to Bone
        avatar.bones.RightArm.quaternion.slerp(rig.RightArm.q, 0.5);
        ```
        

### For Unity (C#)

Use the **Animation Rigging Package**.

1. Create a `TwoBoneIKConstraint` on the avatar's arms.
    
2. Create empty GameObjects: `RightHandTarget`, `RightElbowHint`.
    
3. Script:
    
    ```
    void Update() {
        // Map ML coordinates to Unity World Space
        Vector3 mlHandPos = GetMLPrediction(); 
    
        // Apply to Target (the IK system handles the rotations automatically)
        target.position = Vector3.Lerp(target.position, mlHandPos, Time.deltaTime * smoothSpeed);
    }
    ```
    

## 5. Summary of Recommendations

1. **Format:** Train your model to output **MediaPipe Holistic (World)** coordinates (meters), NOT screen-space (pixels).
    
2. **Solver:** Use **Kalidokit** (JS) or **Animation Rigging** (Unity) to handle the math. Don't write raw quaternion math from scratch if you don't have to.
    
3. **Smoothing:** Implement **1€ Filter** immediately. It is essential for professional feel.