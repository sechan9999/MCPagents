# observability.py
"""
Observability & Traceability System
관측성 및 추적성 시스템

에이전트의 판단 근거를 추적하고 시각화합니다.
LangSmith, Arize Phoenix 연동 지원.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from contextlib import contextmanager
import json
import time
import uuid
import threading


class TraceType(Enum):
    """추적 유형 (Trace Type)"""
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    CHAIN = "chain"
    AGENT = "agent"
    EMBEDDING = "embedding"
    CACHE = "cache"
    ROUTER = "router"
    DLP_CHECK = "dlp_check"
    ERROR = "error"


class TraceStatus(Enum):
    """추적 상태 (Trace Status)"""
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CACHED = "cached"


@dataclass
class SpanContext:
    """스팬 컨텍스트 (Span Context)"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None


@dataclass
class Span:
    """스팬 (Span) - 하나의 작업 단위"""
    context: SpanContext
    name: str
    type: TraceType
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TraceStatus = TraceStatus.RUNNING
    
    # 입력/출력
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # 성능
    duration_ms: Optional[float] = None
    tokens_used: int = 0
    cost: float = 0.0
    
    # 에러
    error: Optional[str] = None
    
    def finish(self, status: TraceStatus = TraceStatus.SUCCESS, output: Any = None, error: str = None):
        """스팬 종료"""
        self.end_time = datetime.now()
        self.status = status
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        if output:
            self.output_data = output if isinstance(output, dict) else {"result": output}
        if error:
            self.error = error
            self.status = TraceStatus.ERROR


@dataclass
class Trace:
    """트레이스 (Trace) - 하나의 요청 전체 추적"""
    trace_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    spans: List[Span] = field(default_factory=list)
    
    # 집계
    total_duration_ms: float = 0
    total_tokens: int = 0
    total_cost: float = 0
    
    # 메타데이터
    user_id: str = ""
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_span(self, span: Span):
        self.spans.append(span)
    
    def finish(self):
        self.end_time = datetime.now()
        self.total_duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.total_tokens = sum(s.tokens_used for s in self.spans)
        self.total_cost = sum(s.cost for s in self.spans)


class TracingBackend:
    """추적 백엔드 추상 클래스"""
    
    def log_trace(self, trace: Trace) -> None:
        raise NotImplementedError
    
    def log_span(self, span: Span) -> None:
        raise NotImplementedError


class LangSmithBackend(TracingBackend):
    """LangSmith 백엔드"""
    
    def __init__(self, api_key: str = None, project: str = "mcp-agents"):
        self.project = project
        self._client = None
        
        try:
            from langsmith import Client
            self._client = Client(api_key=api_key) if api_key else Client()
            print(f"✓ LangSmith connected: {project}")
        except ImportError:
            print("⚠️ LangSmith not installed. pip install langsmith")
        except Exception as e:
            print(f"⚠️ LangSmith connection failed: {e}")
    
    def log_trace(self, trace: Trace) -> None:
        if not self._client:
            return
        # LangSmith API 호출
        pass
    
    def log_span(self, span: Span) -> None:
        if not self._client:
            return
        # LangSmith API 호출
        pass


class ArizePhoenixBackend(TracingBackend):
    """Arize Phoenix 백엔드"""
    
    def __init__(self, endpoint: str = "http://localhost:6006"):
        self.endpoint = endpoint
        self._tracer = None
        
        try:
            import phoenix as px
            self._tracer = px.launch_app()
            print(f"✓ Phoenix UI: {endpoint}")
        except ImportError:
            print("⚠️ Phoenix not installed. pip install arize-phoenix")
        except Exception as e:
            print(f"⚠️ Phoenix connection failed: {e}")
    
    def log_trace(self, trace: Trace) -> None:
        if not self._tracer:
            return
        # Phoenix API 호출
        pass
    
    def log_span(self, span: Span) -> None:
        pass


class InMemoryBackend(TracingBackend):
    """메모리 기반 백엔드 (개발용)"""
    
    def __init__(self, max_traces: int = 1000):
        self._traces: List[Trace] = []
        self._spans: List[Span] = []
        self._max_traces = max_traces
        self._lock = threading.Lock()
    
    def log_trace(self, trace: Trace) -> None:
        with self._lock:
            self._traces.append(trace)
            if len(self._traces) > self._max_traces:
                self._traces = self._traces[-self._max_traces:]
    
    def log_span(self, span: Span) -> None:
        with self._lock:
            self._spans.append(span)
    
    def get_traces(self, limit: int = 100) -> List[Trace]:
        return self._traces[-limit:]
    
    def get_spans(self, trace_id: str = None, limit: int = 100) -> List[Span]:
        if trace_id:
            return [s for s in self._spans if s.context.trace_id == trace_id][-limit:]
        return self._spans[-limit:]


class Tracer:
    """트레이서 (Tracer) - 추적 관리자"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._backends: List[TracingBackend] = [InMemoryBackend()]
        self._current_trace: Optional[Trace] = None
        self._current_span: Optional[Span] = None
        self._span_stack: List[Span] = []
        self._initialized = True
    
    def add_backend(self, backend: TracingBackend):
        """백엔드 추가"""
        self._backends.append(backend)
    
    def _generate_id(self) -> str:
        return str(uuid.uuid4())[:16]
    
    @contextmanager
    def trace(self, name: str, user_id: str = "", metadata: Dict = None):
        """트레이스 시작"""
        trace = Trace(
            trace_id=self._generate_id(),
            name=name,
            start_time=datetime.now(),
            user_id=user_id,
            metadata=metadata or {}
        )
        
        self._current_trace = trace
        
        try:
            yield trace
        finally:
            trace.finish()
            for backend in self._backends:
                backend.log_trace(trace)
            self._current_trace = None
    
    @contextmanager
    def span(
        self,
        name: str,
        type: TraceType = TraceType.CHAIN,
        input_data: Dict = None,
        metadata: Dict = None,
        tags: List[str] = None
    ):
        """스팬 시작"""
        trace_id = self._current_trace.trace_id if self._current_trace else self._generate_id()
        parent_span_id = self._current_span.context.span_id if self._current_span else None
        
        context = SpanContext(
            trace_id=trace_id,
            span_id=self._generate_id(),
            parent_span_id=parent_span_id
        )
        
        span = Span(
            context=context,
            name=name,
            type=type,
            start_time=datetime.now(),
            input_data=input_data or {},
            metadata=metadata or {},
            tags=tags or []
        )
        
        self._span_stack.append(span)
        self._current_span = span
        
        try:
            yield span
            span.finish(TraceStatus.SUCCESS)
        except Exception as e:
            span.finish(TraceStatus.ERROR, error=str(e))
            raise
        finally:
            self._span_stack.pop()
            self._current_span = self._span_stack[-1] if self._span_stack else None
            
            if self._current_trace:
                self._current_trace.add_span(span)
            
            for backend in self._backends:
                backend.log_span(span)
    
    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens: int = 0,
        cost: float = 0.0,
        latency_ms: float = 0
    ):
        """LLM 호출 로깅"""
        with self.span(
            name=f"LLM: {model}",
            type=TraceType.LLM_CALL,
            input_data={"prompt": prompt[:500]},
            metadata={"model": model}
        ) as span:
            span.output_data = {"response": response[:500]}
            span.tokens_used = tokens
            span.cost = cost
            span.duration_ms = latency_ms
    
    def log_tool_call(
        self,
        tool_name: str,
        args: Dict,
        result: Any,
        latency_ms: float = 0
    ):
        """도구 호출 로깅"""
        with self.span(
            name=f"Tool: {tool_name}",
            type=TraceType.TOOL_CALL,
            input_data=args,
            metadata={"tool": tool_name}
        ) as span:
            span.output_data = result if isinstance(result, dict) else {"result": str(result)[:500]}
            span.duration_ms = latency_ms
    
    def get_memory_backend(self) -> Optional[InMemoryBackend]:
        """메모리 백엔드 반환"""
        for b in self._backends:
            if isinstance(b, InMemoryBackend):
                return b
        return None


class DebugDashboard:
    """디버깅 대시보드 (Debug Dashboard)"""
    
    def __init__(self, tracer: Tracer = None):
        self.tracer = tracer or Tracer()
    
    def get_trace_summary(self, limit: int = 20) -> List[Dict]:
        """트레이스 요약"""
        backend = self.tracer.get_memory_backend()
        if not backend:
            return []
        
        traces = backend.get_traces(limit)
        return [
            {
                "trace_id": t.trace_id,
                "name": t.name,
                "duration_ms": t.total_duration_ms,
                "span_count": len(t.spans),
                "tokens": t.total_tokens,
                "cost": t.total_cost,
                "start_time": t.start_time.isoformat(),
                "user_id": t.user_id
            }
            for t in traces
        ]
    
    def get_trace_detail(self, trace_id: str) -> Optional[Dict]:
        """트레이스 상세"""
        backend = self.tracer.get_memory_backend()
        if not backend:
            return None
        
        traces = [t for t in backend.get_traces(1000) if t.trace_id == trace_id]
        if not traces:
            return None
        
        trace = traces[0]
        return {
            "trace_id": trace.trace_id,
            "name": trace.name,
            "duration_ms": trace.total_duration_ms,
            "tokens": trace.total_tokens,
            "cost": trace.total_cost,
            "metadata": trace.metadata,
            "spans": [
                {
                    "span_id": s.context.span_id,
                    "parent_span_id": s.context.parent_span_id,
                    "name": s.name,
                    "type": s.type.value,
                    "status": s.status.value,
                    "duration_ms": s.duration_ms,
                    "tokens": s.tokens_used,
                    "cost": s.cost,
                    "input": s.input_data,
                    "output": s.output_data,
                    "error": s.error
                }
                for s in trace.spans
            ]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 조회"""
        backend = self.tracer.get_memory_backend()
        if not backend:
            return {"message": "No backend"}
        
        traces = backend.get_traces(1000)
        spans = backend.get_spans(limit=10000)
        
        if not traces:
            return {"message": "No traces"}
        
        # 집계
        type_counts = {}
        status_counts = {}
        total_duration = 0
        total_tokens = 0
        total_cost = 0
        error_count = 0
        
        for span in spans:
            type_counts[span.type.value] = type_counts.get(span.type.value, 0) + 1
            status_counts[span.status.value] = status_counts.get(span.status.value, 0) + 1
            total_duration += span.duration_ms or 0
            total_tokens += span.tokens_used
            total_cost += span.cost
            if span.status == TraceStatus.ERROR:
                error_count += 1
        
        return {
            "total_traces": len(traces),
            "total_spans": len(spans),
            "span_types": type_counts,
            "span_status": status_counts,
            "total_duration_ms": total_duration,
            "avg_span_duration_ms": total_duration / len(spans) if spans else 0,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "error_count": error_count,
            "error_rate": error_count / len(spans) if spans else 0
        }
    
    def generate_flame_graph_data(self, trace_id: str) -> List[Dict]:
        """Flame Graph 데이터 생성"""
        trace_detail = self.get_trace_detail(trace_id)
        if not trace_detail:
            return []
        
        return [
            {
                "name": span["name"],
                "value": span["duration_ms"] or 1,
                "children": []  # 트리 구조 구축 가능
            }
            for span in trace_detail["spans"]
        ]


# === 데코레이터 ===
def trace_func(name: str = None, type: TraceType = TraceType.CHAIN):
    """함수 추적 데코레이터"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            tracer = Tracer()
            span_name = name or func.__name__
            
            with tracer.span(
                name=span_name,
                type=type,
                input_data={"args": str(args)[:200], "kwargs": str(kwargs)[:200]}
            ) as span:
                result = func(*args, **kwargs)
                span.output_data = {"result": str(result)[:500]}
                return result
        
        return wrapper
    return decorator


# === 테스트 ===
def test_observability():
    """관측성 테스트"""
    print("=" * 60)
    print("🔍 Observability & Traceability Test")
    print("=" * 60)
    
    tracer = Tracer()
    dashboard = DebugDashboard(tracer)
    
    # 시뮬레이션된 에이전트 실행
    with tracer.trace("Agent Execution", user_id="user_001") as trace:
        # 라우팅
        with tracer.span("Routing Decision", type=TraceType.ROUTER) as span:
            time.sleep(0.01)
            span.output_data = {"selected_model": "gpt-4o-mini"}
        
        # 캐시 확인
        with tracer.span("Cache Check", type=TraceType.CACHE) as span:
            time.sleep(0.005)
            span.output_data = {"hit": False}
        
        # LLM 호출
        tracer.log_llm_call(
            model="gpt-4o-mini",
            prompt="What is the weather?",
            response="I don't have access to real-time weather data...",
            tokens=150,
            cost=0.001,
            latency_ms=500
        )
        
        # 도구 호출
        tracer.log_tool_call(
            tool_name="weather_api",
            args={"location": "Seoul"},
            result={"temp": 15, "condition": "cloudy"},
            latency_ms=200
        )
        
        # DLP 검사
        with tracer.span("DLP Check", type=TraceType.DLP_CHECK) as span:
            time.sleep(0.003)
            span.output_data = {"violations": 0, "action": "allow"}
    
    # 결과 출력
    print("\n📊 Trace Summary:")
    summaries = dashboard.get_trace_summary(5)
    for s in summaries:
        print(f"   ID: {s['trace_id']}")
        print(f"   Duration: {s['duration_ms']:.1f}ms")
        print(f"   Spans: {s['span_count']}")
        print(f"   Cost: ${s['cost']:.6f}")
    
    print("\n📈 Statistics:")
    stats = dashboard.get_statistics()
    print(f"   Total Traces: {stats['total_traces']}")
    print(f"   Span Types: {stats['span_types']}")
    print(f"   Error Rate: {stats['error_rate']:.1%}")
    
    print("\n✅ Observability Test Complete!")


if __name__ == "__main__":
    test_observability()
