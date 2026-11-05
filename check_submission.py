"""B 지역 배치표 형식 검사기 — 서버에 올리기 전에 돌린다.  python check_submission.py 우리팀_배치표.csv
검사: 서버와 같은 읽기 규칙(첫 열 머리글 · 빈칸 · 숫자 · 정수 · 같은 대여소 두 번) · 대여소 전부 · D01~D30 전부 ·
0 이상 정수 · 거치대 수 이하 · 매일 합계가 같음(= 총 자전거 수 N) · N 은 1대 이상. 표준 라이브러리만 쓴다.
"""
import csv
import io
import math
import sys

# 한국어 Windows(cp949) 콘솔·파이프·Jupyter 에서 ✓·✗ 를 찍다 멈추지 않게 출력은 UTF-8 로(10-30 검토)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

CAPS = {"B151": 18, "B100": 15, "B075": 30, "B059": 15, "B142": 13, "B131": 20, "B147": 35, "B084": 46, "B064": 12, "B031": 13, "B039": 10, "B082": 21, "B093": 10, "B140": 15, "B144": 12, "B025": 10, "B002": 12, "B152": 20, "B134": 20, "B097": 26, "B095": 19, "B155": 20, "B126": 10, "B154": 25, "B057": 13, "B027": 10, "B139": 20, "B124": 10, "B033": 10, "B121": 10, "B145": 20, "B094": 10, "B079": 10, "B125": 15, "B141": 20, "B137": 14, "B009": 14, "B083": 16, "B046": 10, "B130": 10, "B045": 10, "B102": 10, "B108": 10, "B052": 15, "B070": 14, "B056": 17, "B050": 15, "B116": 12, "B034": 15, "B107": 11, "B117": 10, "B156": 10, "B068": 10, "B060": 10, "B148": 20, "B048": 10, "B132": 10, "B032": 10, "B018": 15, "B036": 10, "B081": 8, "B053": 10, "B028": 15, "B051": 16, "B077": 14, "B150": 12, "B157": 10, "B110": 10, "B153": 10, "B086": 12, "B146": 14, "B123": 10, "B076": 15, "B073": 10, "B049": 20, "B001": 10, "B118": 25, "B061": 10, "B136": 9, "B005": 15, "B041": 10, "B069": 10, "B089": 10, "B129": 15, "B017": 12, "B011": 15, "B109": 10, "B010": 10, "B063": 15, "B012": 10, "B013": 20, "B014": 10, "B071": 10, "B096": 10, "B054": 10, "B047": 10, "B101": 10, "B040": 10, "B043": 10, "B004": 10, "B113": 10, "B106": 15, "B133": 10, "B091": 10, "B128": 9, "B092": 10, "B074": 15, "B088": 12, "B055": 12, "B021": 12, "B003": 12, "B085": 8, "B058": 6, "B042": 10, "B022": 20, "B099": 14, "B143": 15, "B080": 9, "B112": 15, "B090": 10, "B072": 10, "B127": 15, "B008": 7, "B029": 10, "B098": 26, "B020": 31, "B078": 10, "B149": 10, "B015": 10, "B087": 10, "B115": 8, "B065": 10, "B044": 6, "B023": 10, "B114": 10, "B006": 13, "B062": 7, "B138": 10, "B158": 10, "B024": 20, "B007": 8, "B104": 7, "B030": 7, "B016": 7, "B159": 8, "B067": 10, "B035": 10, "B038": 12, "B120": 7, "B122": 10, "B111": 11, "B037": 30, "B103": 7, "B026": 10, "B119": 10, "B135": 15, "B105": 7, "B019": 10, "B066": 7}
MIN_TOTAL = 1  # 총 자전거 수 N 은 양의 정수
MAX_TOTAL = sum(CAPS.values())  # 보유 상한 = 거치대 합
DAYS = ["D%02d" % d for d in range(1, 31)]
# 서버 채점 코드와 같은 머리글 모음 — station_no 는 서버가 받지 않는다
STATION_HEADERS = {"대여소", "대여소번호", "대여소 id", "대여소id", "station", "station_id", "id"}
DAY_HEADERS = {"날짜", "day", "date", "평가일", "일자"}
COUNT_HEADERS = {"배치대수", "배치", "대수", "count", "alloc", "x", "bikes"}


class FormatError(Exception):
    pass


def day_code(h):
    """'26' · 'Day26' · 'D26' · 'd26' → 'D26'. 못 알아보면 대문자 원문."""
    s = (h or "").strip().upper().replace(" ", "")
    for p in ("DAY", "D"):
        if s.startswith(p):
            s = s[len(p):]
            break
    return "D%02d" % int(s) if s.isdigit() else (h or "").strip().upper()


def to_int(raw, where):
    """빈칸 · 숫자 아님(nan·inf 포함) · 정수 아님 → FormatError. 음수는 통과시키고 뒤에서 '0 이상' 으로 잡는다."""
    raw = raw.strip()
    if raw == "":
        raise FormatError("%s 값이 비어 있습니다." % where)
    try:
        f = float(raw)
    except ValueError:
        raise FormatError("%s 값 '%s' 은 숫자가 아닙니다." % (where, raw))
    if not math.isfinite(f):
        raise FormatError("%s 값 '%s' 은 숫자가 아닙니다." % (where, raw))
    if abs(f - round(f)) > 1e-9:
        raise FormatError("%s 값 %s 은 정수가 아닙니다." % (where, raw))
    return int(round(f))


def read(path):
    """CSV → ({날짜: {대여소: 대수}}, 읽기 오류 목록). 파일 자체를 못 읽으면 FormatError."""
    try:
        raw = open(path, "rb").read()
    except OSError as e:
        raise FormatError("파일을 열 수 없습니다: %s (%s)" % (path, e.strerror or e))
    text = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise FormatError("CSV 인코딩을 읽을 수 없습니다 — UTF-8 로 저장해 주세요.")
    rows = [r for r in csv.reader(io.StringIO(text.lstrip("﻿"))) if any(c.strip() for c in r)]
    if not rows:
        raise FormatError("배치표가 비어 있습니다.")
    head = [h.strip().lstrip("﻿").lower() for h in rows[0]]
    out = {d: {} for d in DAYS}
    errs = []
    if any(h in DAY_HEADERS for h in head) and any(h in COUNT_HEADERS for h in head):  # 세로형
        try:
            si = next(i for i, h in enumerate(head) if h in STATION_HEADERS)
            di = next(i for i, h in enumerate(head) if h in DAY_HEADERS)
            ci = next(i for i, h in enumerate(head) if h in COUNT_HEADERS)
        except StopIteration:
            raise FormatError("배치표 머리글은 'Day,대여소 ID,배치대수' 꼴이어야 합니다.")
        for ln, r in enumerate(rows[1:], 2):
            if len(r) <= max(si, di, ci):
                errs.append("%d행의 열 수가 부족합니다." % ln)
                continue
            sid, d = r[si].strip(), day_code(r[di])
            if d not in out:
                errs.append("%d행의 날짜 '%s' 는 D01~D30(1~30)이 아닙니다." % (ln, r[di]))
                continue
            try:
                v = to_int(r[ci], "%d행 %s 의 %s" % (ln, sid, d))
            except FormatError as e:
                errs.append(str(e))
                continue
            if sid in out[d]:
                errs.append("%s 의 %s 가 두 번 나옵니다(%d행)." % (d, sid, ln))
                continue
            out[d][sid] = v
        return out, errs
    if head[0] not in STATION_HEADERS:  # 가로형
        raise FormatError("첫 열 머리글은 'station_id' 여야 합니다(제출틀_배치표_B지역.csv 와 같은 꼴. "
                          "또는 세로형 'Day,대여소 ID,배치대수'). 지금 첫 열 머리글: '%s'" % rows[0][0].strip())
    cols = {}
    for i, h in enumerate(head[1:], 1):
        if day_code(h) in out:
            cols[day_code(h)] = i
    miss = [d for d in DAYS if d not in cols]
    if miss:
        raise FormatError("열 %s 이(가) 없습니다. 머리글은 station_id,D01,...,D30 입니다." % ", ".join(miss[:5]))
    seen = set()
    for ln, r in enumerate(rows[1:], 2):
        sid = r[0].strip()
        if not sid:
            continue
        if sid in seen:
            errs.append("%s 가 두 번 나옵니다(%d행)." % (sid, ln))
            continue
        seen.add(sid)
        for d in DAYS:
            i = cols[d]
            try:
                out[d][sid] = to_int(r[i] if i < len(r) else "", "%d행 %s 의 %s" % (ln, sid, d))
            except FormatError as e:
                errs.append(str(e))
    return out, errs


def check(alloc):
    """서버 검사(validate_alloc)와 같은 규칙·같은 문장."""
    errs = []
    totals = []  # (날짜, 합계) — 값 오류가 없는 날만
    too_few = []
    for d in DAYS:
        col = alloc[d]
        miss = [s for s in CAPS if s not in col]
        extra = [s for s in col if s not in CAPS]
        if miss:
            errs.append("%s: 대여소 %d곳의 배치가 비어 있습니다 (예: %s)." % (d, len(miss), ", ".join(miss[:5])))
        if extra:
            errs.append("%s: 모르는 대여소 %s 가 있습니다." % (d, ", ".join(sorted(extra)[:5])))
        bad = [s for s, v in col.items() if v < 0]
        if bad:
            errs.append("%s: 배치 수량은 0 이상의 정수여야 합니다 (예: %s)." % (d, ", ".join(sorted(bad)[:5])))
            continue
        tot = sum(col.values())
        if tot > MAX_TOTAL:
            errs.append("%s: 합계가 %d대로 보유 상한 %d대를 넘습니다." % (d, tot, MAX_TOTAL))
        elif tot < MIN_TOTAL:
            too_few.append((d, tot))
        totals.append((d, tot))
        over = [s for s in CAPS if s in col and col[s] > CAPS[s]]
        if over:
            errs.append("%s: 거치대 수를 넘는 대여소 %d곳 (예: %s)." % (d, len(over), ", ".join("%s %d>%d" % (s, col[s], CAPS[s]) for s in over[:5])))
    kinds = {t for _, t in totals}
    if len(kinds) > 1:
        listed = ", ".join("%s %d" % kv for kv in totals[:8]) + (", …" if len(totals) > 8 else "")
        errs.append("매일 합계가 같아야 합니다(= 총 자전거 수). 날짜별 합계: %s (서로 다른 합계 %d가지)." % (listed, len(kinds)))
    if too_few:
        listed = ", ".join("%s %d" % kv for kv in too_few[:8]) + (", …" if len(too_few) > 8 else "")
        errs.append("총 자전거 수(하루 배치 합계)는 %d대 이상이어야 합니다. 합계가 모자란 날 %d일: %s" % (MIN_TOTAL, len(too_few), listed))
    return errs, (totals[0][1] if totals else 0)


def fail(errs):
    print("✗ %d건" % len(errs))
    for e in errs[:30]:
        print("  -", e)
    if len(errs) > 30:
        print("  - … 나머지 %d건" % (len(errs) - 30))
    return 1


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    try:
        alloc, errs = read(sys.argv[1])
    except FormatError as e:
        return fail([str(e)])
    if errs:
        return fail(errs)
    errs, n = check(alloc)
    if errs:
        return fail(errs)
    print("✓ 통과 — B 지역 %d곳 × 30일, 총 자전거 수 N = %d대" % (len(CAPS), n))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # 생각 못 한 파일도 traceback 대신 한 줄로
        print("✗ 1건")
        print("  - 검사기가 이 파일을 읽지 못했습니다: %s" % e)
        sys.exit(1)
