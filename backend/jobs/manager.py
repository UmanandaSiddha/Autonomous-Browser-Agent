import asyncio
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.db.models import AutomationJob, JobStatus, JobStep


# asyncio only keeps weak references to running tasks, so a
# background automation can be garbage collected mid-run unless
# something holds on to it.
_background_tasks: set[asyncio.Task] = set()


class JobManager:

    def create_job(
        self,
        db: Session,
        user_id: str,
        automation_type: str,
    ) -> AutomationJob:

        job = AutomationJob(
            id=f"job_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            automation_type=automation_type,
            status=JobStatus.QUEUED,
            step=JobStep.QUEUED,
            progress=0,
        )

        db.add(job)
        db.commit()
        db.refresh(job)

        print(
            f"[JOB] Created {job.id}"
            f" for user {user_id}"
        )

        return job

    def update_job(
        self,
        db: Session,
        job_id: str,
        *,
        status: JobStatus | None = None,
        step: JobStep | None = None,
        progress: int | None = None,
        error: str | None = None,
        result: dict | None = None,
    ) -> AutomationJob:

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
            job.result = result

        job.updated_at = (
            datetime.now(timezone.utc)
        )

        db.commit()
        db.refresh(job)

        print(
            f"[JOB] {job.id} -> "
            f"status={job.status.value}, "
            f"step={job.step.value if job.step else None}, "
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

        _background_tasks.add(task)

        task.add_done_callback(
            _background_tasks.discard
        )

        return task


job_manager = JobManager()
