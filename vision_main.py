from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import List, Optional
import math
import datetime
from sqlalchemy import create_engine, Column, BigInteger, Integer, Float, String, Text, DateTime, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from apscheduler.schedulers.background import BackgroundScheduler
import contextlib


def get_kst_now():
    return datetime.datetime.utcnow() + datetime.timedelta(hours=9)


DATABASE_URL = "mysql+pymysql://root:0215@127.0.0.1:3306/turtlely_db?charset=utf8mb4"

# MySQL 엔진 내부의 일시적인 메타데이터 캐싱 및 sql_mode 제약을 완화하는 옵션 주입
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # 매 요청마다 연결 상태 및 최신 스펙 확인
    pool_recycle=1800,        # 커넥션 자동 갱신 (좀비 세션 방지)
    connect_args={"init_command": "SET SESSION sql_mode='STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION'"}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Member(Base):
    __tablename__ = "member"

    member_id = Column("member_id", BigInteger, primary_key=True, index=True, autoincrement=True)
    login_id = Column("login_id", String(50), nullable=True)
    password = Column(String(255), nullable=True)
    nickname = Column(String(50), nullable=True)  
    
    social_type = Column("social_type", String(255), nullable=True)
    
    phone_number = Column("phone_number", String(255), nullable=True)
    social_id = Column("social_id", String(255), nullable=True)
    
    role = Column(String(255), nullable=True)


class MonthlyMeasurement(Base):
    __tablename__ = "monthly_measurement"

    monthly_id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    member_id = Column(BigInteger, ForeignKey("member.member_id"), nullable=False)

    created_at = Column(DateTime, nullable=False, default=get_kst_now)
    updated_at = Column(DateTime, nullable=False, default=get_kst_now, onupdate=get_kst_now)

    cva_angle = Column(Float, nullable=True)
    cra_angle = Column(Float, nullable=True)
    posture_type = Column(String(255), nullable=True)
    measured_at = Column(DateTime, nullable=True, default=get_kst_now)
    score = Column(Integer, nullable=True)

    predicted_diseases = Column(Text, nullable=True)
    prediction_data = Column(Text, nullable=True)


class Notification(Base):
    __tablename__ = "notification_tb"

    __table_args__ = {'extend_existing': True}

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


class MonthlyReportResponse(BaseModel):
    status: int
    message: str
    dataStatus: str
    year: int
    month: int
    nickname: str
    postureType: str
    posture_message: Optional[str] = None
    cvaAngle: Optional[float] = None
    craAngle: Optional[float] = None
    total_measurements: int
    isAlarmSet: bool
    availableDate: Optional[str] = None


class NotificationReportRequest(BaseModel):
    member_id: int


class NotificationData(BaseModel):
    nickname: str
    isAlarmSet: bool


class NotificationReportResponse(BaseModel):
    status: int
    message: str
    data: NotificationData


class GraphElement(BaseModel):
    month: str
    cvaAngle: float
    craAngle: float


class GraphReportResponse(BaseModel):
    status: int
    message: str
    dataStatus: str
    nickname: str
    graphData: List[GraphElement]


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
    scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    scheduler.add_job(check_30day_remeasure, "cron", hour=9, minute=0)
    scheduler.start()
    print("30일 정기 재측정 자동 알림 스케줄러가 활성화되었습니다.")
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "errorCode": "INVALID_INPUT_TYPE",
            "message": "파라미터에 잘못된 타입의 값이 전달되었습니다."
        }
    )


@app.get("/")
def read_root():
    return {"message": "Turtly Vision AI Server is Running!"}


@app.post("/report/analyze")
async def analyze_posture(data: AnalyzeRequest, db: Session = Depends(get_db)):
    try:
        member = db.query(Member).filter(Member.member_id == data.member_id).first()
        if not member:
            return JSONResponse(status_code=404, content={"status": "error", "message": "존재하지 않는 회원입니다."})

        now_time = get_kst_now()
        new_report = MonthlyMeasurement(
            member_id=member.member_id,
            created_at=now_time,
            updated_at=now_time,
            measured_at=now_time,
            posture_type="분석 중",
            prediction_data="AI가 자세를 분석하고 있습니다. 잠시만 기다려주세요.",
            cva_angle=None,
            cra_angle=None,
            score=None
        )
        db.add(new_report)
        db.commit()
        db.refresh(new_report)

        if not data.frames:
            db.delete(new_report)
            db.commit()
            return {"status": "error", "message": "전송된 프레임 데이터가 없습니다."}

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
            db.delete(new_report)
            db.commit()
            return {"status": "error", "message": "랜드마크를 확실하게 찾을 수 없습니다."}

        delta_y_cva = abs(best_frame.c7_y - best_frame.tragus_y)
        delta_x_cva = abs(best_frame.c7_x - best_frame.tragus_x)
        cva_angle = math.degrees(math.atan2(delta_y_cva, delta_x_cva))

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

        if cva_angle >= 50:
            status = "정상"
            calculated_score = 100
            msg = "정상입니다."
        elif 45 <= cva_angle < 50:
            status = "일자목"
            calculated_score = 80
            msg = "일자목 단계입니다." if cra_angle > 155 else "경추의 C자 곡선이 펴지고 있습니다."
        elif 40 <= cva_angle < 45:
            status = "거북목"
            calculated_score = 60
            msg = "거북목 상태입니다."
        else:
            status = "역C자목"
            calculated_score = 40
            msg = "전문적인 교정과 진단을 권장합니다."

        new_report.cva_angle = round(cva_angle, 2)
        new_report.cra_angle = round(cra_angle, 2)
        new_report.posture_type = status
        new_report.score = calculated_score
        new_report.prediction_data = msg
        new_report.updated_at = get_kst_now()

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


@app.get("/report/monthly", response_model=MonthlyReportResponse)
def get_monthly_report(member_id: int, year: int, month: int, db: Session = Depends(get_db)):
    try:
        kst_now = get_kst_now()

        if month < 1 or month > 12 or year > kst_now.year or (year == kst_now.year and month > kst_now.month):
            return JSONResponse(
                status_code=400,
                content={
                    "errorCode": "INVALID_DATE_RANGE",
                    "message": "유효하지 않은 날짜 범위입니다."
                }
            )
        member = db.query(Member).filter(Member.member_id == member_id).first()
        if not member:
            return JSONResponse(
                status_code=404,
                content={
                    "errorCode": "MEMBER_NOT_FOUND",
                    "message": "존재하지 않는 회원입니다."
                }
            )

        target_member_id = int(member.member_id)

        is_alarm_set = db.query(Notification).filter(
            Notification.member_id == target_member_id,
            Notification.type == "MONTHLY",
            Notification.status == "PENDING"
        ).first() is not None

        last_total_report = db.query(MonthlyMeasurement).filter(
            MonthlyMeasurement.member_id == target_member_id,
            MonthlyMeasurement.posture_type != "분석 중",
            MonthlyMeasurement.posture_type.isnot(None)
        ).order_by(MonthlyMeasurement.measured_at.desc()).first()

        start_date = datetime.datetime(year, month, 1)
        end_date = datetime.datetime(year + 1, 1, 1) if month == 12 else datetime.datetime(year, month + 1, 1)

        monthly_reports = db.query(MonthlyMeasurement).filter(
            MonthlyMeasurement.member_id == target_member_id,
            MonthlyMeasurement.measured_at >= start_date,
            MonthlyMeasurement.measured_at < end_date
        ).order_by(MonthlyMeasurement.measured_at.desc()).all()

        if monthly_reports and (
            monthly_reports[0].posture_type == "분석 중" or
            monthly_reports[0].cva_angle is None
        ):
            return {
                "status": 200,
                "message": "현재 리포트 데이터를 분석 및 산출 중입니다.",
                "dataStatus": "PROCESSING",
                "year": year,
                "month": month,
                "nickname": member.nickname,
                "postureType": "분석 중",
                "posture_message": "AI가 자세를 분석하고 있습니다. 잠시만 기다려주세요.",
                "cvaAngle": None,
                "craAngle": None,
                "total_measurements": 0,
                "isAlarmSet": is_alarm_set,
                "availableDate": None
            }

        if not monthly_reports:
            base_date = last_total_report.measured_at if last_total_report else kst_now
            available_date_str = (base_date + datetime.timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")

            return {
                "status": 200,
                "message": "정기 측정 기록이 존재하지 않습니다.",
                "dataStatus": "NOT_YET",
                "year": year,
                "month": month,
                "nickname": member.nickname,
                "postureType": "데이터 없음",
                "posture_message": "이번 달 측정 기록이 존재하지 않습니다. 검사를 진행해 주세요.",
                "cvaAngle": None,
                "craAngle": None,
                "total_measurements": 0,
                "isAlarmSet": is_alarm_set,
                "availableDate": available_date_str
            }

        report = monthly_reports[0]

        return {
            "status": 200,
            "message": "정기 검사 리포트 조회가 완료되었습니다.",
            "dataStatus": "AVAILABLE",
            "year": year,
            "month": month,
            "nickname": member.nickname,
            "postureType": report.posture_type or "미정",
            "posture_message": report.prediction_data or "측정 데이터가 존재합니다.",
            "cvaAngle": report.cva_angle,
            "craAngle": report.cra_angle,
            "total_measurements": len(monthly_reports),
            "isAlarmSet": is_alarm_set,
            "availableDate": None
        }

    except SQLAlchemyError as db_err:
        return JSONResponse(
            status_code=500,
            content={
                "errorCode": "DATABASE_ERROR",
                "message": f"DB 오류: {str(db_err)}"
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "errorCode": "SERVER_INTERNAL_ERROR",
                "message": str(e)
            }
        )


@app.post("/notification/report", response_model=NotificationReportResponse)
def apply_report_notification(req: NotificationReportRequest, db: Session = Depends(get_db)):
    try:
        member = db.query(Member).filter(Member.member_id == req.member_id).first()
        if not member:
            return JSONResponse(
                status_code=404,
                content={
                    "errorCode": "MEMBER_NOT_FOUND",
                    "message": f"요청된 member_id '{req.member_id}'에 해당하는 사용자가 존재하지 않습니다."
                }
            )

        target_member_id = int(member.member_id)

        existing = db.query(Notification).filter(
            Notification.member_id == target_member_id,
            Notification.type == "MONTHLY",
            Notification.status == "PENDING"
        ).first()

        if not existing:
            new_alert = Notification(
                member_id=target_member_id,
                type="MONTHLY",
                content="AI 자세 분석 리포트가 성공적으로 산출 완료되었습니다!",
                status="PENDING",
                sent_at=None
            )
            db.add(new_alert)
            db.commit()

        return {
            "status": 200,
            "message": "분석 완료 알림 신청이 성공적으로 접수되었습니다.",
            "data": {
                "nickname": member.nickname,
                "isAlarmSet": True
            }
        }

    except SQLAlchemyError as db_err:
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={
                "errorCode": "DATABASE_ERROR",
                "message": f"알림 제약조건 위반 또는 데이터 무결성 오류 발생: {str(db_err)}"
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "errorCode": "SERVER_INTERNAL_ERROR",
                "message": str(e)
            }
        )


@app.get("/report/graph", response_model=GraphReportResponse)
def get_monthly_graph_data(member_id: int, db: Session = Depends(get_db)):
    try:
        member = db.query(Member).filter(Member.member_id == member_id).first()
        if not member:
            return JSONResponse(
                status_code=404,
                content={
                    "errorCode": "MEMBER_NOT_FOUND",
                    "message": "요청된 member_id에 해당하는 사용자가 존재하지 않습니다."
                }
            )

        target_member_id = int(member.member_id)

        all_measurements = db.query(MonthlyMeasurement).filter(
            MonthlyMeasurement.member_id == target_member_id,
            MonthlyMeasurement.posture_type != "분석 중",
            MonthlyMeasurement.posture_type.isnot(None),
            MonthlyMeasurement.cva_angle.isnot(None),
            MonthlyMeasurement.cra_angle.isnot(None),
            MonthlyMeasurement.measured_at.isnot(None)
        ).order_by(MonthlyMeasurement.measured_at.asc()).all()

        if not all_measurements:
            return {
                "status": 200,
                "message": "측정된 그래프 기록이 존재하지 않습니다.",
                "dataStatus": "EMPTY",
                "nickname": member.nickname,
                "graphData": []
            }

        monthly_map = {}
        for row in all_measurements:
            month_key = f"{row.measured_at.year}-{row.measured_at.month}"
            
            monthly_map[month_key] = {
                "month_label": f"{row.measured_at.month}월",
                "cvaAngle": row.cva_angle,
                "craAngle": row.cra_angle,
                "raw_date": row.measured_at  
            }

        sorted_graph_data = sorted(monthly_map.values(), key=lambda x: x["raw_date"])

        final_graph_list = [
            GraphElement(
                month=item["month_label"],
                cvaAngle=item["cvaAngle"],
                craAngle=item["craAngle"]
            )
            for item in sorted_graph_data
        ]

        return {
            "status": 200,
            "message": "월간 각도 변화 그래프 데이터 조회가 완료되었습니다.",
            "dataStatus": "AVAILABLE",
            "nickname": member.nickname,
            "graphData": final_graph_list
        }

    except SQLAlchemyError as db_err:
        return JSONResponse(
            status_code=500,
            content={
                "errorCode": "DATABASE_ERROR",
                "message": f"데이터베이스 연결 및 그래프 쿼리 수행 도중 오류 발생: {str(db_err)}"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "errorCode": "SERVER_INTERNAL_ERROR",
                "message": f"서버 내부 데이터 연산 실패: {str(e)}"
            }
        )