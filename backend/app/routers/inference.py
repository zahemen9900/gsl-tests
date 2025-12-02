from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.inference_seq2seq import Seq2SeqInferenceService

router = APIRouter()
_inference_service = Seq2SeqInferenceService()


@router.websocket("/ws/inference")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _inference_service.reset()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "frame" in payload:
                frame = payload["frame"]
                transcript = _inference_service.process_frame(frame)
                if transcript:
                    await websocket.send_json({"transcript": transcript})
            elif payload.get("command") == "reset":
                _inference_service.reset()
                await websocket.send_json({"status": "reset"})
    except WebSocketDisconnect:
        _inference_service.reset()
