from fastapi import APIRouter, HTTPException

from backend.jobs.email_worker import (
    run_email_automation,
)
from backend.jobs.manager import job_manager


router = APIRouter(
    prefix="/api/automations",
    tags=["Automations"],
)


@router.post("/email")
async def start_email_automation():

    print(
        "[API][AUTOMATION] "
        "Email automation requested"
    )

    job = job_manager.create_job()

    await job_manager.run_in_background(
        job.id,
        run_email_automation,
    )

    return {
        "job_id": job.id,
        "status": job.status,
    }


@router.get("/email/{job_id}")
async def get_email_automation(
    job_id: str,
):

    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Automation job not found",
        )

    return job