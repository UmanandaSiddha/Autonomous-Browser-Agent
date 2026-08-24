import asyncio
import uuid
from datetime import datetime, timezone

from backend.jobs.models import (
    AutomationJob,
    JobStatus,
    JobStep,
)


class JobManager:

    def __init__(self):
        self.jobs: dict[str, AutomationJob] = {}
        self.tasks: dict[str, asyncio.Task] = {}

    def create_job(self) -> AutomationJob:
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        now = datetime.now(timezone.utc)

        job = AutomationJob(
            id=job_id,
            status=JobStatus.QUEUED,
            step=JobStep.INITIALIZING,
            progress=0,
            created_at=now,
            updated_at=now,
        )

        self.jobs[job_id] = job

        print(
            f"[JOB] Created job {job_id}"
        )

        return job

    def get_job(
        self,
        job_id: str,
    ) -> AutomationJob | None:

        return self.jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        **updates,
    ) -> AutomationJob | None:

        job = self.jobs.get(job_id)

        if job is None:
            return None

        updated = job.model_copy(
            update={
                **updates,
                "updated_at": datetime.now(
                    timezone.utc
                ),
            }
        )

        self.jobs[job_id] = updated

        print(
            f"[JOB] {job_id} → "
            f"status={updated.status.value}, "
            f"step={updated.step.value}, "
            f"progress={updated.progress}%"
        )

        return updated

    async def run_in_background(
        self,
        job_id: str,
        worker,
    ):
        """
        Run an automation without blocking
        the HTTP request.
        """

        task = asyncio.create_task(
            worker(job_id)
        )

        self.tasks[job_id] = task

        return task


job_manager = JobManager()