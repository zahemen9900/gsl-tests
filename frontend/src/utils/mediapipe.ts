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

// MediaPipe pose connections for skeleton drawing
const POSE_CONNECTIONS: [number, number][] = [
    // Face
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    // Torso
    [9, 10], [11, 12], [11, 23], [12, 24], [23, 24],
    // Left arm
    [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    // Right arm
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    // Left leg
    [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
    // Right leg
    [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];

// Hand connections
const HAND_CONNECTIONS: [number, number][] = [
    // Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    // Index
    [0, 5], [5, 6], [6, 7], [7, 8],
    // Middle
    [0, 9], [9, 10], [10, 11], [11, 12],
    // Ring
    [0, 13], [13, 14], [14, 15], [15, 16],
    // Pinky
    [0, 17], [17, 18], [18, 19], [19, 20],
    // Palm
    [5, 9], [9, 13], [13, 17],
];

// Colors for different body parts
const COLORS = {
    pose: { line: 'rgba(99, 102, 241, 0.8)', point: 'rgba(139, 92, 246, 1)' },
    leftHand: { line: 'rgba(34, 197, 94, 0.8)', point: 'rgba(74, 222, 128, 1)' },
    rightHand: { line: 'rgba(249, 115, 22, 0.8)', point: 'rgba(251, 146, 60, 1)' },
    face: { line: 'rgba(236, 72, 153, 0.4)', point: 'rgba(244, 114, 182, 0.6)' },
};

/**
 * Draw landmarks and skeleton connections on canvas
 */
export function drawLandmarks(
    ctx: CanvasRenderingContext2D,
    results: Results,
    width: number,
    height: number
): void {
    ctx.save();
    // Mirror the canvas to match the mirrored webcam
    ctx.translate(width, 0);
    ctx.scale(-1, 1);

    // Draw pose skeleton
    if (results.poseLandmarks) {
        drawConnections(ctx, results.poseLandmarks, POSE_CONNECTIONS, width, height, COLORS.pose.line, 3);
        drawPoints(ctx, results.poseLandmarks, width, height, COLORS.pose.point, 6);
    }

    // Draw left hand
    if (results.leftHandLandmarks) {
        drawConnections(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, width, height, COLORS.leftHand.line, 2);
        drawPoints(ctx, results.leftHandLandmarks, width, height, COLORS.leftHand.point, 4);
    }

    // Draw right hand
    if (results.rightHandLandmarks) {
        drawConnections(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, width, height, COLORS.rightHand.line, 2);
        drawPoints(ctx, results.rightHandLandmarks, width, height, COLORS.rightHand.point, 4);
    }

    // Draw face mesh (simplified - only key points)
    if (results.faceLandmarks) {
        // Draw only key facial landmarks for better performance
        const keyFaceIndices = [
            // Eyes
            33, 133, 362, 263,
            // Eyebrows
            70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
            // Nose
            1, 4, 5, 195, 197,
            // Mouth
            61, 291, 0, 17, 78, 308,
            // Face outline
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ];
        const keyLandmarks = keyFaceIndices.map(i => results.faceLandmarks![i]).filter(Boolean);
        drawPoints(ctx, keyLandmarks, width, height, COLORS.face.point, 2);
    }

    ctx.restore();
}

function drawConnections(
    ctx: CanvasRenderingContext2D,
    landmarks: NormalizedLandmark[],
    connections: [number, number][],
    width: number,
    height: number,
    color: string,
    lineWidth: number
): void {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const [start, end] of connections) {
        const startLm = landmarks[start];
        const endLm = landmarks[end];
        if (!startLm || !endLm) continue;

        // Skip low-visibility landmarks for cleaner visualization
        const startVis = (startLm as any).visibility ?? 1;
        const endVis = (endLm as any).visibility ?? 1;
        if (startVis < 0.5 || endVis < 0.5) continue;

        ctx.beginPath();
        ctx.moveTo(startLm.x * width, startLm.y * height);
        ctx.lineTo(endLm.x * width, endLm.y * height);
        ctx.stroke();
    }
}

function drawPoints(
    ctx: CanvasRenderingContext2D,
    landmarks: NormalizedLandmark[],
    width: number,
    height: number,
    color: string,
    radius: number
): void {
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = radius;

    for (const lm of landmarks) {
        if (!lm) continue;
        const vis = (lm as any).visibility ?? 1;
        if (vis < 0.5) continue;

        ctx.beginPath();
        ctx.arc(lm.x * width, lm.y * height, radius, 0, 2 * Math.PI);
        ctx.fill();
    }

    ctx.shadowBlur = 0;
}

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
