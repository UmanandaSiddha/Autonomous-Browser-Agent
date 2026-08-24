import asyncio
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.db.models import AutomationJob

# from backend.jobs.models import (
#     AutomationJob,
#     JobStatus,
#     JobStep,
# )


class JobManager:

    # def __init__(self):
    #     self.jobs: dict[str, AutomationJob] = {}
    #     self.tasks: dict[str, asyncio.Task] = {}

    def create_job(
        self,
        db: Session,
        user_id: str,
        automation_type: str,
    ) -> AutomationJob:
        # job_id = f"job_{uuid.uuid4().hex[:12]}"

        # now = datetime.now(timezone.utc)

        # job = AutomationJob(
        #     id=job_id,
        #     status=JobStatus.QUEUED,
        #     step=JobStep.INITIALIZING,
        #     progress=0,
        #     created_at=now,
        #     updated_at=now,
        # )

        job = AutomationJob(
            id=f"job_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            automation_type=automation_type,
            status="queued",
            step="queued",
            progress=0,
        )

        db.add(job)
        db.commit()
        db.refresh(job)

        print(
            f"[JOB] Created {job.id}"
            f" for user {user_id}"
        )

        # self.jobs[job_id] = job

        # print(
        #     f"[JOB] Created job {job_id}"
        # )

        return job

    def get_job(
        self,
        db: Session,
        job_id: str,
    ) -> AutomationJob | None:

        job = db.get(
            AutomationJob,
            job_id,
        )

        if job is None:
            return None

        return job

    def update_job(
        self,
        # job_id: str,
        # **updates,
        db: Session,
        job_id: str,
        *,
        status: str | None = None,
        step: str | None = None,
        progress: int | None = None,
        error: str | None = None,
        result: dict | None = None,
    ) -> AutomationJob | None:

        job = db.get(
            AutomationJob,
            job_id,
        )

        if job is None:
            raise ValueError(
                f"Job {job_id} not found"
            )

        if status is not None:
            job.status = status

        if step is not None:
            job.step = step

        if progress is not None:
            job.progress = progress

        if error is not None:
            job.error = error

        if result is not None:
            job.result = json.dumps(
                result
            )

        job.updated_at = (
            datetime.now(timezone.utc)
        )

        db.commit()
        db.refresh(job)

        print(
            f"[JOB] {job.id} → "
            f"status={job.status}, "
            f"step={job.step}, "
            f"progress={job.progress}%"
        )

        return job

    async def run_in_background(
        self,
        job_id: str,
        user_id: str,
        worker,
    ):
        """
        Run an automation without blocking
        the HTTP request.
        """

        task = asyncio.create_task(
            worker(job_id, user_id)
        )

        # self.tasks[job_id] = task

        return task


job_manager = JobManager()