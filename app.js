// ============================================================
// FRAME — Hand-Gesture Mask Camera Web App
// 100% client-side, zero build step
// ============================================================

import {
    HandLandmarker,
    FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- DOM refs ----
const video = document.getElementById("input");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");
const loadScreen = document.getElementById("loading-screen");
const loadText = document.getElementById("loading-text");
const errorScreen = document.getElementById("error-screen");
const errorTitle = document.getElementById("error-title");
const errorMsg = document.getElementById("error-message");
const statusPill = document.getElementById("status-pill");
const statusText = document.getElementById("status-text");
const shutterBtn = document.getElementById("shutter-btn");
const recordBtn = document.getElementById("record-btn");
const flipBtn = document.getElementById("flip-btn");
const flashEl = document.getElementById("shutter-flash");
const timerEl = document.getElementById("rec-timer");
const galleryImg = document.getElementById("gallery-img");
const effectBtns = document.querySelectorAll(".effect-btn");

// ---- State ----
let handLandmarker = null;
let stream = null;
let facingMode = "user";
let animFrameId = null;
let lastVideoTime = -1;

// Smoothed polygon points (4 corners)
let smoothedPoints = null;
const LERP_FACTOR = 0.3;

// Recording state
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let recStartTime = 0;
let recTimerInterval = null;

// Current effect
let currentEffect = "pop";

// Off-screen canvas for pixel manipulation effects
const offCanvas = document.createElement("canvas");
const offCtx = offCanvas.getContext("2d", { willReadFrequently: true });

// ============================================================
// 1. CAMERA
// ============================================================

async function startCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
    }
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: true,
        });
        video.srcObject = stream;
        await new Promise(r => (video.onloadedmetadata = r));
        video.play();
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        offCanvas.width = canvas.width;
        offCanvas.height = canvas.height;
    } catch (err) {
        showError(err);
        throw err;
    }
}

function showError(err) {
    loadScreen.classList.add("hidden");
    errorScreen.classList.add("visible");
    const msg = err.message || String(err);
    if (msg.includes("Permission") || msg.includes("denied")) {
        errorTitle.textContent = "Camera Access Denied";
        errorMsg.textContent =
            "Please allow camera access in your browser settings and reload the page.";
    } else if (msg.includes("NotFoundError") || msg.includes("no camera")) {
        errorTitle.textContent = "No Camera Found";
        errorMsg.textContent =
            "We couldn't find a camera on this device. Please connect one and try again.";
    } else {
        errorTitle.textContent = "Something Went Wrong";
        errorMsg.textContent = msg;
    }
}

flipBtn.addEventListener("click", async () => {
    facingMode = facingMode === "user" ? "environment" : "user";
    await startCamera();
});

// ============================================================
// 2. MEDIAPIPE HAND LANDMARKER
// ============================================================

async function initHandTracker() {
    loadText.textContent = "Loading hand-tracking model…";
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath:
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU",
            },
            runningMode: "VIDEO",
            numHands: 2,
        });
    } catch (err) {
        showError(err);
        throw err;
    }
}

// ============================================================
// 3. GESTURE LOGIC — 4-POINT POLYGON FROM BOTH HANDS
// ============================================================

// MediaPipe landmark indices
const THUMB_TIP = 4;
const INDEX_TIP = 8;

function landmarkToCanvas(lm) {
    // Mirror x for selfie view when using front camera
    const x = facingMode === "user" ? (1 - lm.x) * canvas.width : lm.x * canvas.width;
    const y = lm.y * canvas.height;
    return { x, y };
}

/**
 * Given two hands' landmarks, extract 4 points:
 *   left thumb tip, left index tip, right index tip, right thumb tip
 * and order them as a proper convex quadrilateral.
 */
function computeShapePoints(results) {
    if (!results || !results.landmarks || results.landmarks.length < 2) {
        return null;
    }

    const hand0 = results.landmarks[0];
    const hand1 = results.landmarks[1];

    // Get wrist positions to determine which hand is left/right on screen
    const wrist0 = landmarkToCanvas(hand0[0]);
    const wrist1 = landmarkToCanvas(hand1[0]);

    let leftHand, rightHand;
    if (wrist0.x < wrist1.x) {
        leftHand = hand0;
        rightHand = hand1;
    } else {
        leftHand = hand1;
        rightHand = hand0;
    }

    // 4 corner points
    const lt = landmarkToCanvas(leftHand[THUMB_TIP]);   // left thumb
    const li = landmarkToCanvas(leftHand[INDEX_TIP]);    // left index
    const ri = landmarkToCanvas(rightHand[INDEX_TIP]);   // right index
    const rt = landmarkToCanvas(rightHand[THUMB_TIP]);   // right thumb

    // Order as convex quadrilateral: sort by angle from centroid
    const points = [lt, li, ri, rt];
    const cx = (lt.x + li.x + ri.x + rt.x) / 4;
    const cy = (lt.y + li.y + ri.y + rt.y) / 4;

    points.sort((a, b) => {
        return Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx);
    });

    // Reject very small shapes (< 8% of shortest canvas side)
    const minDim = Math.min(canvas.width, canvas.height);
    const threshold = minDim * 0.08;
    const maxDist = Math.max(
        dist(points[0], points[2]),
        dist(points[1], points[3])
    );
    if (maxDist < threshold) return null;

    return points;
}

function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function lerpPoint(a, b, t) {
    return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

function smoothPoints(raw) {
    if (!smoothedPoints) {
        smoothedPoints = raw.map(p => ({ ...p }));
    } else {
        for (let i = 0; i < 4; i++) {
            smoothedPoints[i] = lerpPoint(smoothedPoints[i], raw[i], LERP_FACTOR);
        }
    }
    return smoothedPoints;
}

// ============================================================
// 4. EFFECTS — Applied INSIDE the polygon shape
// ============================================================

function drawEffectedRegion(points) {
    switch (currentEffect) {
        case "pop": drawPopEffect(points); break;
        case "blur": drawBlurEffect(points); break;
        case "invert": drawInvertEffect(points); break;
        case "thermal": drawThermalEffect(points); break;
        case "pixelate": drawPixelateEffect(points); break;
        default: drawPopEffect(points); break;
    }
}

/** Pop: high saturation + contrast boost inside the shape */
function drawPopEffect(points) {
    ctx.save();
    buildClipPath(points);
    ctx.filter = "saturate(2.2) contrast(1.3) brightness(1.1)";
    drawMirroredVideo();
    ctx.filter = "none";
    ctx.restore();
}

/** Blur: heavy gaussian blur inside the shape */
function drawBlurEffect(points) {
    ctx.save();
    buildClipPath(points);
    ctx.filter = "blur(16px)";
    drawMirroredVideo();
    ctx.filter = "none";
    ctx.restore();
}

/** Invert: colour inversion inside the shape */
function drawInvertEffect(points) {
    ctx.save();
    buildClipPath(points);
    ctx.filter = "invert(1)";
    drawMirroredVideo();
    ctx.filter = "none";
    ctx.restore();
}

/** Thermal: simulated thermal/infrared look */
function drawThermalEffect(points) {
    // Draw into off-screen canvas, then pixel-manipulate
    offCtx.save();
    if (facingMode === "user") {
        offCtx.translate(offCanvas.width, 0);
        offCtx.scale(-1, 1);
    }
    offCtx.drawImage(video, 0, 0, offCanvas.width, offCanvas.height);
    offCtx.restore();

    // Get bounding box of the polygon for perf
    const bb = getBoundingBox(points);
    const x = Math.max(0, Math.floor(bb.x));
    const y = Math.max(0, Math.floor(bb.y));
    const w = Math.min(Math.ceil(bb.w), offCanvas.width - x);
    const h = Math.min(Math.ceil(bb.h), offCanvas.height - y);

    if (w <= 0 || h <= 0) return;

    const imageData = offCtx.getImageData(x, y, w, h);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        // Map grayscale → thermal gradient (blue→cyan→green→yellow→red→white)
        const t = gray / 255;
        if (t < 0.2) {
            data[i] = 0; data[i + 1] = 0; data[i + 2] = lerp(80, 255, t / 0.2);
        } else if (t < 0.4) {
            const s = (t - 0.2) / 0.2;
            data[i] = 0; data[i + 1] = lerp(0, 255, s); data[i + 2] = lerp(255, 0, s);
        } else if (t < 0.6) {
            const s = (t - 0.4) / 0.2;
            data[i] = lerp(0, 255, s); data[i + 1] = 255; data[i + 2] = 0;
        } else if (t < 0.8) {
            const s = (t - 0.6) / 0.2;
            data[i] = 255; data[i + 1] = lerp(255, 0, s); data[i + 2] = 0;
        } else {
            const s = (t - 0.8) / 0.2;
            data[i] = 255; data[i + 1] = lerp(0, 255, s); data[i + 2] = lerp(0, 255, s);
        }
    }
    offCtx.putImageData(imageData, x, y);

    // Clip and draw the thermal image
    ctx.save();
    buildClipPath(points);
    ctx.drawImage(offCanvas, 0, 0);
    ctx.restore();
}

/** Pixelate: mosaic effect inside the shape */
function drawPixelateEffect(points) {
    const bb = getBoundingBox(points);
    const blockSize = 12;

    offCtx.save();
    if (facingMode === "user") {
        offCtx.translate(offCanvas.width, 0);
        offCtx.scale(-1, 1);
    }
    offCtx.drawImage(video, 0, 0, offCanvas.width, offCanvas.height);
    offCtx.restore();

    const x = Math.max(0, Math.floor(bb.x));
    const y = Math.max(0, Math.floor(bb.y));
    const w = Math.min(Math.ceil(bb.w), offCanvas.width - x);
    const h = Math.min(Math.ceil(bb.h), offCanvas.height - y);

    if (w <= 0 || h <= 0) return;

    const imageData = offCtx.getImageData(x, y, w, h);
    const data = imageData.data;

    for (let by = 0; by < h; by += blockSize) {
        for (let bx = 0; bx < w; bx += blockSize) {
            // Average the block
            let rSum = 0, gSum = 0, bSum = 0, count = 0;
            for (let dy = 0; dy < blockSize && by + dy < h; dy++) {
                for (let dx = 0; dx < blockSize && bx + dx < w; dx++) {
                    const idx = ((by + dy) * w + (bx + dx)) * 4;
                    rSum += data[idx]; gSum += data[idx + 1]; bSum += data[idx + 2];
                    count++;
                }
            }
            const rAvg = rSum / count, gAvg = gSum / count, bAvg = bSum / count;
            // Fill the block
            for (let dy = 0; dy < blockSize && by + dy < h; dy++) {
                for (let dx = 0; dx < blockSize && bx + dx < w; dx++) {
                    const idx = ((by + dy) * w + (bx + dx)) * 4;
                    data[idx] = rAvg; data[idx + 1] = gAvg; data[idx + 2] = bAvg;
                }
            }
        }
    }
    offCtx.putImageData(imageData, x, y);

    ctx.save();
    buildClipPath(points);
    ctx.drawImage(offCanvas, 0, 0);
    ctx.restore();
}

function lerp(a, b, t) { return a + (b - a) * t; }

function getBoundingBox(points) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of points) {
        if (p.x < minX) minX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.x > maxX) maxX = p.x;
        if (p.y > maxY) maxY = p.y;
    }
    return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

function buildClipPath(points) {
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.closePath();
    ctx.clip();
}

function drawMirroredVideo() {
    if (facingMode === "user") {
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}

// ============================================================
// 5. NEON BORDER + CORNER BRACKETS
// ============================================================

function drawPolygonOverlay(points) {
    const accentColor = getEffectAccent();

    // Glow border
    ctx.save();
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 3;
    ctx.shadowColor = accentColor;
    ctx.shadowBlur = 18;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.restore();

    // Corner dots
    ctx.save();
    for (const p of points) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = accentColor;
        ctx.shadowColor = accentColor;
        ctx.shadowBlur = 12;
        ctx.fill();
        // Inner white dot
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = "#fff";
        ctx.shadowBlur = 0;
        ctx.fill();
    }
    ctx.restore();

    // Corner brackets (L-shaped)
    const bracketLen = 18;
    ctx.save();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    for (let i = 0; i < points.length; i++) {
        const curr = points[i];
        const prev = points[(i - 1 + points.length) % points.length];
        const next = points[(i + 1) % points.length];

        // Direction towards previous and next
        const toPrev = normalize({ x: prev.x - curr.x, y: prev.y - curr.y });
        const toNext = normalize({ x: next.x - curr.x, y: next.y - curr.y });

        ctx.beginPath();
        ctx.moveTo(curr.x + toPrev.x * bracketLen, curr.y + toPrev.y * bracketLen);
        ctx.lineTo(curr.x, curr.y);
        ctx.lineTo(curr.x + toNext.x * bracketLen, curr.y + toNext.y * bracketLen);
        ctx.stroke();
    }
    ctx.restore();
}

function normalize(v) {
    const len = Math.hypot(v.x, v.y) || 1;
    return { x: v.x / len, y: v.y / len };
}

function getEffectAccent() {
    switch (currentEffect) {
        case "pop": return "#00e5ff";
        case "blur": return "#a78bfa";
        case "invert": return "#f472b6";
        case "thermal": return "#ff6b2b";
        case "pixelate": return "#34d399";
        default: return "#00e5ff";
    }
}

// ============================================================
// 6. RENDER LOOP
// ============================================================

function renderLoop() {
    animFrameId = requestAnimationFrame(renderLoop);

    if (video.readyState < 2) return;

    // --- Draw normal video as base layer ---
    ctx.save();
    if (facingMode === "user") {
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    // --- Hand detection ---
    if (!handLandmarker) return;

    const nowMs = performance.now();
    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;

    let results;
    try {
        results = handLandmarker.detectForVideo(video, nowMs);
    } catch {
        return;
    }

    // --- Shape from 4 points ---
    const rawPoints = computeShapePoints(results);

    if (rawPoints) {
        const points = smoothPoints(rawPoints);
        drawEffectedRegion(points);

        // Update HUD
        statusPill.classList.add("active");
        statusText.textContent = `${currentEffect.toUpperCase()} — Shape active`;
    } else {
        // Decay smoothed points so next appearance doesn't snap from old position
        smoothedPoints = null;
        statusPill.classList.remove("active");
        statusText.textContent = "Show both hands ✌️";
    }
}

// ============================================================
// 7. PHOTO CAPTURE
// ============================================================

// 5-second countdown timer
const countdownEl = document.getElementById("countdown");
let countdownActive = false;

shutterBtn.addEventListener("click", () => {
    if (countdownActive) return;
    countdownActive = true;
    let remaining = 5;
    countdownEl.textContent = remaining;
    countdownEl.classList.add("visible");

    const tick = setInterval(() => {
        remaining--;
        if (remaining > 0) {
            countdownEl.textContent = remaining;
        } else {
            clearInterval(tick);
            countdownEl.classList.remove("visible");
            capturePhoto();
            countdownActive = false;
        }
    }, 1000);
});

function capturePhoto() {
    // Flash effect
    flashEl.classList.add("flash");
    setTimeout(() => flashEl.classList.remove("flash"), 120);

    canvas.toBlob(blob => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);

        // Trigger download
        const a = document.createElement("a");
        a.href = url;
        a.download = `frame-${Date.now()}.png`;
        a.click();

        // Show in gallery thumbnail
        galleryImg.src = url;
        galleryImg.style.display = "block";
    }, "image/png");
}

// ============================================================
// 8. VIDEO RECORDING
// ============================================================

recordBtn.addEventListener("click", () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

function startRecording() {
    const canvasStream = canvas.captureStream(30);

    // Add mic audio track if available
    if (stream) {
        const audioTracks = stream.getAudioTracks();
        audioTracks.forEach(t => canvasStream.addTrack(t));
    }

    // Pick a supported mimeType
    const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]
        .find(m => MediaRecorder.isTypeSupported(m)) || "";

    recordedChunks = [];
    mediaRecorder = new MediaRecorder(canvasStream, mimeType ? { mimeType } : {});

    mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "video/webm" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = `frame-${Date.now()}.webm`;
        a.click();

        // Gallery thumbnail from current canvas frame
        canvas.toBlob(tb => {
            if (tb) {
                galleryImg.src = URL.createObjectURL(tb);
                galleryImg.style.display = "block";
            }
        });
    };

    mediaRecorder.start(100);
    isRecording = true;
    recordBtn.classList.add("recording");
    recStartTime = Date.now();
    timerEl.classList.add("visible");
    statusPill.classList.add("recording");
    updateRecTimer();
    recTimerInterval = setInterval(updateRecTimer, 500);
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    isRecording = false;
    recordBtn.classList.remove("recording");
    timerEl.classList.remove("visible");
    statusPill.classList.remove("recording");
    clearInterval(recTimerInterval);
}

function updateRecTimer() {
    const elapsed = Math.floor((Date.now() - recStartTime) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const s = String(elapsed % 60).padStart(2, "0");
    timerEl.textContent = `${m}:${s}`;
}

// ============================================================
// 9. EFFECT SELECTOR
// ============================================================

effectBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        effectBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentEffect = btn.dataset.effect;
    });
});

// ============================================================
// 10. GALLERY THUMBNAIL — open last capture in new tab
// ============================================================

document.getElementById("gallery-thumb").addEventListener("click", () => {
    if (galleryImg.src && galleryImg.style.display !== "none") {
        window.open(galleryImg.src, "_blank");
    }
});

// ============================================================
// INIT
// ============================================================

async function init() {
    try {
        loadText.textContent = "Requesting camera access…";
        await startCamera();
        await initHandTracker();
        loadScreen.classList.add("hidden");
        renderLoop();
    } catch (err) {
        console.error("Init failed:", err);
        // Error screen is already shown by showError()
    }
}

init();
