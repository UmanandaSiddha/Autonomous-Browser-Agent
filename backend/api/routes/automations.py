from fastapi import APIRouter, HTTPException, Depends

from sqlalchemy.orm import Session

from backend.jobs.email_worker import (
    run_email_automation,
)
from backend.jobs.manager import job_manager

from backend.auth.dependencies import get_current_user
from backend.db.database import get_db
from backend.db.models import AutomationJob, User


router = APIRouter(
    prefix="/api/automations",
    tags=["Automations"],
)


@router.post("/email")
async def start_email_automation(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    print(
        "[API][AUTOMATION] "
        "Email automation requested"
    )

    job = job_manager.create_job(
        db=db,
        user_id=current_user.id,
        automation_type="email",
    )

    await job_manager.run_in_background(
        job.id,
        current_user.id,
        run_email_automation,
    )

    return {
        "job_id": job.id,
        "status": job.status,
    }


@router.get("/email/{job_id}")
async def get_email_automation(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    job = (
        db.query(AutomationJob)
        .filter(
            AutomationJob.id == job_id,
            AutomationJob.user_id == current_user.id,
        )
        .first()
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Automation job not found",
        )

    return job