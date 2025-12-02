import { Holistic, Results, NormalizedLandmark } from '@mediapipe/holistic';

const FACE_DOWNSAMPLE = 5;
const NUM_POSE_LANDMARKS = 33;
const NUM_HAND_LANDMARKS = 21;
const NUM_FACE_LANDMARKS = Math.ceil(468 / FACE_DOWNSAMPLE); // 94
const PER_FRAME_FEATURES = NUM_POSE_LANDMARKS * 4 + 2 * NUM_HAND_LANDMARKS * 3 + NUM_FACE_LANDMARKS * 3; // 540

const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_HIP = 23;
const RIGHT_HIP = 24;

function normalizeFrameLandmarks(feats: Float32Array): Float32Array {
    // feats is a flat array of 540 elements
    // Structure: Pose(132) | LeftHand(63) | RightHand(63) | Face(282)
    
    const poseOffset = 0;
    const leftHandOffset = NUM_POSE_LANDMARKS * 4;
    const rightHandOffset = leftHandOffset + NUM_HAND_LANDMARKS * 3;
    const faceOffset = rightHandOffset + NUM_HAND_LANDMARKS * 3;

    // Extract Torso points for normalization
    // Pose structure: x, y, z, v
    const getPosePt = (idx: number) => {
        const base = poseOffset + idx * 4;
        return { x: feats[base], y: feats[base + 1], z: feats[base + 2] };
    };

    const ls = getPosePt(LEFT_SHOULDER);
    const rs = getPosePt(RIGHT_SHOULDER);
    const lh = getPosePt(LEFT_HIP);
    const rh = getPosePt(RIGHT_HIP);

    // Check validity (simple check: if all are 0, likely invalid)
    if (ls.x === 0 && ls.y === 0 && ls.z === 0) return feats;

    const torsoCenter = {
        x: (ls.x + rs.x + lh.x + rh.x) / 4,
        y: (ls.y + rs.y + lh.y + rh.y) / 4,
        z: (ls.z + rs.z + lh.z + rh.z) / 4
    };

    const dist = (p1: any, p2: any) => Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2) + Math.pow(p1.z - p2.z, 2));

    const shoulderSpan = dist(ls, rs);
    const hipSpan = dist(lh, rh);
    const torsoHeight = dist(
        { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2, z: (ls.z + rs.z) / 2 },
        { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2, z: (lh.z + rh.z) / 2 }
    );

    const candidates = [shoulderSpan, hipSpan, torsoHeight].filter(v => v > 1e-4);
    const scale = candidates.length > 0 ? candidates.sort((a, b) => a - b)[Math.floor(candidates.length / 2)] : 1.0;

    // Apply normalization
    const normalized = new Float32Array(feats.length);
    
    // Helper to normalize a point (x,y,z)
    const normPt = (x: number, y: number, z: number) => {
        return {
            x: (x - torsoCenter.x) / scale,
            y: (y - torsoCenter.y) / scale,
            z: (z - torsoCenter.z) / scale
        };
    };

    // Pose
    for (let i = 0; i < NUM_POSE_LANDMARKS; i++) {
        const base = poseOffset + i * 4;
        const p = normPt(feats[base], feats[base + 1], feats[base + 2]);
        normalized[base] = p.x;
        normalized[base + 1] = p.y;
        normalized[base + 2] = p.z;
        normalized[base + 3] = feats[base + 3]; // Visibility stays same
    }

    // Left Hand
    for (let i = 0; i < NUM_HAND_LANDMARKS; i++) {
        const base = leftHandOffset + i * 3;
        const p = normPt(feats[base], feats[base + 1], feats[base + 2]);
        normalized[base] = p.x;
        normalized[base + 1] = p.y;
        normalized[base + 2] = p.z;
    }

    // Right Hand
    for (let i = 0; i < NUM_HAND_LANDMARKS; i++) {
        const base = rightHandOffset + i * 3;
        const p = normPt(feats[base], feats[base + 1], feats[base + 2]);
        normalized[base] = p.x;
        normalized[base + 1] = p.y;
        normalized[base + 2] = p.z;
    }

    // Face
    for (let i = 0; i < NUM_FACE_LANDMARKS; i++) {
        const base = faceOffset + i * 3;
        const p = normPt(feats[base], feats[base + 1], feats[base + 2]);
        normalized[base] = p.x;
        normalized[base + 1] = p.y;
        normalized[base + 2] = p.z;
    }

    return normalized;
}

export function extractFeatureVector(results: Results): Float32Array | null {
    const feats = new Float32Array(PER_FRAME_FEATURES);
    let offset = 0;

    // Pose (33 * 4)
    if (results.poseLandmarks) {
        for (const lm of results.poseLandmarks) {
            feats[offset++] = lm.x;
            feats[offset++] = lm.y;
            feats[offset++] = lm.z;
            feats[offset++] = lm.visibility || 0;
        }
    } else {
        offset += NUM_POSE_LANDMARKS * 4;
    }

    // Left Hand (21 * 3)
    if (results.leftHandLandmarks) {
        for (const lm of results.leftHandLandmarks) {
            feats[offset++] = lm.x;
            feats[offset++] = lm.y;
            feats[offset++] = lm.z;
        }
    } else {
        offset += NUM_HAND_LANDMARKS * 3;
    }

    // Right Hand (21 * 3)
    if (results.rightHandLandmarks) {
        for (const lm of results.rightHandLandmarks) {
            feats[offset++] = lm.x;
            feats[offset++] = lm.y;
            feats[offset++] = lm.z;
        }
    } else {
        offset += NUM_HAND_LANDMARKS * 3;
    }

    // Face (94 * 3) - Downsampled
    if (results.faceLandmarks) {
        for (let i = 0; i < 468; i += FACE_DOWNSAMPLE) {
            const lm = results.faceLandmarks[i];
            feats[offset++] = lm.x;
            feats[offset++] = lm.y;
            feats[offset++] = lm.z;
        }
    } else {
        offset += NUM_FACE_LANDMARKS * 3;
    }

    if (offset !== PER_FRAME_FEATURES) {
        console.error(`Feature size mismatch: ${offset} vs ${PER_FRAME_FEATURES}`);
        return null;
    }

    return normalizeFrameLandmarks(feats);
}

let holisticInstancePromise: Promise<Holistic> | null = null;

export function getHolisticInstance(): Promise<Holistic> {
    if (!holisticInstancePromise) {
        holisticInstancePromise = new Promise((resolve, reject) => {
            const holistic = new Holistic({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
            });

            holistic.setOptions({
                modelComplexity: 1,
                smoothLandmarks: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            holistic.initialize()
                .then(() => resolve(holistic))
                .catch((err) => {
                    holisticInstancePromise = null;
                    reject(err);
                });
        });
    }

    return holisticInstancePromise;
}
