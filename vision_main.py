from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import math
import datetime
from sqlalchemy import (
    create_engine,
    Column,
    BigInteger,
    Integer,
    Float,
    String,
    Text,
    DateTime,
    ForeignKey,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from apscheduler.schedulers.background import BackgroundScheduler
import contextlib


def get_kst_now():
    return datetime.datetime.utcnow() + datetime.timedelta(hours=9)


# 데이터베이스 설정
DATABASE_URL = "mysql+pymysql://root:0215@localhost:3306/turtlely_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Member(Base):
    __tablename__ = "member"

    member_id = Column(BigInteger, primary_key=True, index=True)
    login_id = Column(String(50), nullable=True)
    password = Column(String(255), nullable=True)
    nickname = Column(String(50), nullable=True)
    social_type = Column(String(255), nullable=True)
    phone_number = Column(String(255), nullable=True)
    social_id = Column(String(255), nullable=True)
    role = Column(String(255), nullable=True)


class MonthlyMeasurement(Base):
    __tablename__ = "monthly_measurement"

    monthly_id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    member_id = Column(BigInteger, ForeignKey("member.member_id"), nullable=False)

    cva_angle = Column(Float, nullable=False)
    cra_angle = Column(Float, nullable=False)
    posture_type = Column(String(255), nullable=False)
    measured_at = Column(DateTime, nullable=False, default=get_kst_now)
    score = Column(Integer, nullable=False)

    # DB 엔티티 구조를 맞추기 위해 컬럼은 남겨둠
    # 지금 기능에서는 값을 저장하지 않으므로 NULL로 들어감
    predicted_diseases = Column(Text, nullable=True)

    prediction_data = Column(Text, nullable=True)

    # DB 엔티티 구조를 맞추기 위해 컬럼은 남겨둠
    # API 요청에서는 받지 않고, 저장 시에도 넣지 않으므로 NULL로 들어감
    hw_accel_x = Column(Float, nullable=True)
    hw_accel_y = Column(Float, nullable=True)
    hw_accel_z = Column(Float, nullable=True)
    calibration_c = Column(Float, nullable=True)


class Notification(Base):
    __tablename__ = "notification"

    notification_id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    member_id = Column(BigInteger, ForeignKey("member.member_id"), nullable=False)
    type = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    status = Column(String(255), nullable=False)
    sent_at = Column(DateTime, nullable=True)
    read_at = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class FrameData(BaseModel):
    timestamp: str
    eye_x: float
    eye_y: float
    tragus_x: float
    tragus_y: float
    c7_x: float
    c7_y: float


class AnalyzeRequest(BaseModel):
    member_id: int
    frames: List[FrameData]

    class Config:
        json_schema_extra = {
            "example": {
                "member_id": 1,
                "frames": [
                    {
                        "timestamp": "2026-05-28 10:30:00",
                        "eye_x": 0.45,
                        "eye_y": 0.35,
                        "tragus_x": 0.51,
                        "tragus_y": 0.38,
                        "c7_x": 0.50,
                        "c7_y": 0.55
                    }
                ]
            }
        }


class MonthlyReportResponse(BaseModel):
    status: int
    message: str
    year: int
    month: int
    nickname: str
    posture_status: str
    posture_message: Optional[str] = None
    cva_angle: Optional[float] = None
    cra_angle: Optional[float] = None
    total_measurements: int


# 30일 주기 체크 알림
def check_30day_remeasure():
    db: Session = SessionLocal()
    try:
        today = get_kst_now().date()
        target_date = today - datetime.timedelta(days=30)

        subquery = db.query(
            MonthlyMeasurement.member_id,
            func.max(MonthlyMeasurement.measured_at).label("last_measurement")
        ).group_by(MonthlyMeasurement.member_id).subquery()

        users_to_notify = db.query(subquery.c.member_id).filter(
            func.date(subquery.c.last_measurement) == target_date
        ).all()

        for user in users_to_notify:
            notification = Notification(
                member_id=user.member_id,
                type="MONTHLY",
                content="오늘은 정기 재측정 날입니다!",
                status="SENT",
                sent_at=get_kst_now()
            )
            db.add(notification)

        db.commit()
        print(f"[스케줄러] 30일 정기 재측정 대상자 총 {len(users_to_notify)}명 알림 디비 생성 완료.")

    except Exception as e:
        db.rollback()
        print(f"스케줄러 에러 발생: {e}")

    finally:
        db.close()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # 실제 MySQL에서 Spring Boot/JPA가 테이블을 관리한다면 주석 처리 권장 -> 주석처리함(멎나?)
    # Base.metadata.create_all(bind=engine)

    scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    scheduler.add_job(check_30day_remeasure, "cron", hour=9, minute=0)
    scheduler.start()

    print("30일 정기 재측정 자동 알림 스케줄러가 활성화되었습니다. (매일 한국시간 오전 9시 작동)")

    yield

    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Turtly Vision AI Server is Running!"}


# 자세 분석 저장
@app.post("/report/analyze")
async def analyze_posture(data: AnalyzeRequest, db: Session = Depends(get_db)):
    try:
        member = db.query(Member).filter(Member.member_id == data.member_id).first()

        if not member:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "존재하지 않는 회원입니다. 가입 정보를 확인해 주세요."
                }
            )

        if not data.frames:
            return {
                "status": "error",
                "message": "전송된 프레임 데이터가 없습니다."
            }

        best_frame = None
        min_score = float("inf")

        for i, current in enumerate(data.frames):
            current_coords = [
                current.eye_x,
                current.eye_y,
                current.tragus_x,
                current.tragus_y,
                current.c7_x,
                current.c7_y
            ]

            if any(c <= 0 or c > 1.0 for c in current_coords):
                continue

            if i == 0:
                movement = 0.0
            else:
                prev = data.frames[i - 1]
                movement = math.sqrt(
                    (current.tragus_x - prev.tragus_x) ** 2 +
                    (current.tragus_y - prev.tragus_y) ** 2 +
                    (current.c7_x - prev.c7_x) ** 2 +
                    (current.c7_y - prev.c7_y) ** 2
                )

            center_distance = math.sqrt(
                (current.tragus_x - 0.5) ** 2 +
                (current.tragus_y - 0.5) ** 2
            )

            score = (movement * 0.7) + (center_distance * 0.3)

            if score < min_score:
                min_score = score
                best_frame = current

        if not best_frame:
            return {
                "status": "error",
                "message": "머리카락이나 조명으로 인해 목의 랜드마크를 확실하게 찾을 수 없습니다. 장애물을 제거하고 밝은 곳에서 다시 촬영해 주세요.",
                "detail": "No valid frames found after filtering"
            }

        # CVA 계산
        delta_y_cva = abs(best_frame.c7_y - best_frame.tragus_y)
        delta_x_cva = abs(best_frame.c7_x - best_frame.tragus_x)
        cva_angle = math.degrees(math.atan2(delta_y_cva, delta_x_cva))

        # CRA 계산
        v1 = (
            best_frame.eye_x - best_frame.tragus_x,
            best_frame.eye_y - best_frame.tragus_y
        )
        v2 = (
            best_frame.c7_x - best_frame.tragus_x,
            best_frame.c7_y - best_frame.tragus_y
        )

        dot_prod = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 * mag2 != 0:
            cos_theta = dot_prod / (mag1 * mag2)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            cra_angle = math.degrees(math.acos(cos_theta))
        else:
            cra_angle = 0.0

        # 자세 판정
        if cva_angle >= 50:
            status = "정상"
            calculated_score = 100
            msg = "정상입니다."

        elif 45 <= cva_angle < 50:
            status = "일자목"
            calculated_score = 80

            if cra_angle > 155:
                msg = "일자목 단계입니다."
            else:
                msg = "경추의 C자 곡선이 펴지고 있습니다. 틈틈이 스트레칭을 해주세요."

        elif 40 <= cva_angle < 45:
            status = "거북목"
            calculated_score = 60
            msg = "거북목 상태입니다."

        else:
            status = "역C자목"
            calculated_score = 40
            msg = "경추 정렬이 반대로 변형된 위험한 상태입니다. 전문적인 교정과 진단을 권장합니다."

        new_report = MonthlyMeasurement(
            member_id=member.member_id,
            cva_angle=round(cva_angle, 2),
            cra_angle=round(cra_angle, 2),
            posture_type=status,
            measured_at=get_kst_now(),
            score=calculated_score,
            prediction_data=msg
        )

        db.add(new_report)
        db.commit()
        db.refresh(new_report)

        return {
            "status": "success",
            "message": "최적의 프레임 분석 및 결과 저장이 성공적으로 완료되었습니다.",
            "data": {
                "report_id": new_report.monthly_id,
                "posture_status": new_report.posture_type,
                "measured_at": new_report.measured_at.strftime("%Y-%m-%d %H:%M:%S")
            }
        }

    except Exception as e:
        db.rollback()
        return {
            "status": "error",
            "message": "분석 중 오류가 발생했습니다.",
            "detail": str(e)
        }


# 월간 리포트 조회
@app.get("/report/monthly", response_model=MonthlyReportResponse)
def get_monthly_report(member_id: int, year: int, month: int, db: Session = Depends(get_db)):
    try:
        member = db.query(Member).filter(Member.member_id == member_id).first()

        if not member:
            return JSONResponse(
                status_code=404,
                content={
                    "errorCode": "MEMBER_NOT_FOUND",
                    "message": "존재하지 않는 회원입니다."
                }
            )

        start_date = datetime.datetime(year, month, 1)

        if month == 12:
            end_date = datetime.datetime(year + 1, 1, 1)
        else:
            end_date = datetime.datetime(year, month + 1, 1)

        monthly_reports = db.query(MonthlyMeasurement).filter(
            MonthlyMeasurement.member_id == member_id,
            MonthlyMeasurement.measured_at >= start_date,
            MonthlyMeasurement.measured_at < end_date
        ).order_by(MonthlyMeasurement.measured_at.desc()).all()

        if not monthly_reports:
            return {
                "status": 200,
                "message": f"해당 월({year}년 {month}월)의 정기 측정 기록이 존재하지 않습니다.",
                "year": year,
                "month": month,
                "nickname": member.nickname or "회원",
                "posture_status": "데이터 없음",
                "posture_message": "이번 달 측정 기록이 존재하지 않습니다. 검사를 진행해 주세요.",
                "cva_angle": None,
                "cra_angle": None,
                "total_measurements": 0
            }

        report = monthly_reports[0]

        return {
            "status": 200,
            "message": f"{year}년 {month}월 정기 검사 리포트 조회가 완료되었습니다.",
            "year": year,
            "month": month,
            "nickname": member.nickname or "회원",
            "posture_status": report.posture_type,
            "posture_message": report.prediction_data,
            "cva_angle": report.cva_angle,
            "cra_angle": report.cra_angle,
            "total_measurements": 1
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "errorCode": "SERVER_INTERNAL_ERROR",
                "message": f"월간 리포트 가공 중 서버 내부 오류가 발생했습니다: {str(e)}"
            }
        )