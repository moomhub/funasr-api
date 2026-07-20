from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

from src.api import offline, spk, uploads
from src.application.tasks import UploadTooLargeError
from src.core.hotwords import InvalidHotwordFormatError


class _Services:
    def __init__(self, enabled=True, result=None, submission_available=None):
        self.enabled = enabled
        self.submission_available = (
            enabled if submission_available is None else submission_available
        )
        self.result = result or {"ok": True}
        self.calls = []
        self.task_submission_service = SimpleNamespace(
            submit_offline=self._submit_offline,
            submit_speaker=self._submit_speaker,
        )

    def is_engine_enabled(self, mode):
        return self.enabled if mode in {"offline", "spk"} else False

    def is_task_submission_available(self, mode):
        return self.submission_available if mode in {"offline", "spk"} else False

    async def _submit_offline(self, *args, **kwargs):
        self.calls.append(("offline", args, kwargs))
        return self.result

    async def _submit_speaker(self, *args, **kwargs):
        self.calls.append(("spk", args, kwargs))
        return self.result


@pytest.mark.asyncio
async def test_submit_upload_or_error_handles_disabled_mode_before_submission():
    services = _Services(enabled=False)

    response = await uploads.submit_upload_or_error(
        services=services,
        mode="offline",
        disabled_error="OFFLINE 模式未启用",
        submit=lambda: services._submit_offline(),
        logger=offline.logger,
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert services.calls == []


@pytest.mark.asyncio
async def test_submit_upload_rejects_when_runtime_exists_but_queue_is_not_running():
    services = _Services(enabled=True, submission_available=False)

    response = await uploads.submit_upload_or_error(
        services=services,
        mode="offline",
        disabled_error="OFFLINE 模式未启用",
        submit=lambda: services._submit_offline(),
        logger=offline.logger,
    )

    assert response.status_code == 503
    assert services.calls == []


@pytest.mark.asyncio
async def test_submit_upload_or_error_maps_size_and_bad_request_errors():
    services = _Services()

    async def too_large():
        raise UploadTooLargeError("too large")

    async def invalid_hotwords():
        raise InvalidHotwordFormatError("bad hotwords")

    too_large_response = await uploads.submit_upload_or_error(
        services=services,
        mode="offline",
        disabled_error="OFFLINE 模式未启用",
        submit=too_large,
        logger=offline.logger,
        bad_request_errors=(InvalidHotwordFormatError,),
    )
    bad_request_response = await uploads.submit_upload_or_error(
        services=services,
        mode="offline",
        disabled_error="OFFLINE 模式未启用",
        submit=invalid_hotwords,
        logger=offline.logger,
        bad_request_errors=(InvalidHotwordFormatError,),
    )

    assert too_large_response.status_code == 413
    assert bad_request_response.status_code == 400


@pytest.mark.asyncio
async def test_offline_and_spk_routes_delegate_to_shared_upload_helper():
    offline_services = _Services(result={"offline": True})
    spk_services = _Services(result={"spk": True})

    offline_response = await offline.upload_offline_task(
        file=SimpleNamespace(filename="demo.wav"),
        email="user@example.com",
        hotwords=None,
        hotword_id=None,
        vip=True,
        services=offline_services,
    )
    spk_response = await spk.recognize_speaker(
        file=SimpleNamespace(filename="demo.wav"),
        email="user@example.com",
        vip=False,
        services=spk_services,
    )

    assert offline_response == {"offline": True}
    assert spk_response == {"spk": True}
    assert offline_services.calls[0][0] == "offline"
    assert spk_services.calls[0][0] == "spk"


@pytest.mark.asyncio
async def test_spk_task_query_rejects_when_standalone_mode_is_disabled():
    services = _Services(enabled=False)

    response = await spk.get_spk_task("task-1", services=services)

    assert response.status_code == 503
    assert b"SPK" in response.body
