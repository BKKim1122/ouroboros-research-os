# V1–V15 동결 세대 — 자리표시 (placeholder)

> **이 문서는 자리표시자다. V1–V15의 코드·데이터·논문 원본은 아직 이 저장소에
> 포함되지 않았다.** 아래는 원본을 "재구성"한 것이 아니라, 원본이 존재한다는
> 사실과 그 식별정보(파일명·크기·SHA-256)를 정직하게 기록한 것이다.
> 원본을 찾으면 해시로 진위를 대조한 뒤 이 자리에 채운다.

## 왜 비어 있는가

V1–V15는 윈도우 PC에서 수행되어 논문 v4.1로 완결·동결된 세대다. 인덱스 원칙:
"V1–V15 동결. V16 결과로 자동 수정하지 않음." 현재 이 원본 아카이브들이
edgexpert(리눅스) 작업 트리에 없어 미포함 상태다. 요약만 보고 코드/수치를
새로 생성하는 것은 날조가 되므로 하지 않는다.

## 계보 요약 (인덱스 기록 기준, 검증 아님)

- V1–V9: 측정 실패·우회경로·정보소실/성능저하 혼동의 수정
- V10: gain calibration
- V11: 규모·구조·OOD 일반화
- V12: B/S/V 분산 causal coalition
- V13: 내부 사전고정 확증 + generic conditional-routing 경계 확인
- V14: identity attribution과 beneficiary priority의 설계된 이중해리
- V15: relative privilege와 operational concern의 분리
       (privilege 0에서 대칭적 보호와 무관심 구분)

## 원본 아카이브 식별정보 (ASSET_MANIFEST 기준)

찾을 때 아래 파일명으로 검색하고, 찾으면 SHA-256으로 진위 확인.

| 파일 | 크기(byte) | SHA-256 |
|---|---|---|
| identity_privilege_concern_evidence_package_v1.4.zip | 27929221 | f6a618ee…c4fe15bd |
| identity_privilege_concern_final_manuscripts_v4.1.zip | 7582155 | ae87919c…2d4cd753 |
| identity_privilege_concern_submission_kit_v4.1.zip | 7591102 | 11838b35…cdd165ce |
| V16_handoff_package_v1.1.zip | 386882 | eaae4b47…b9122c17 |

(전체 해시는 docs/history/AI_ATTENTION_AWARENESS_LINEAGE.md 및 원본 ASSET_MANIFEST.csv 참조)

## 원본을 찾는 방법

윈도우 PC PowerShell:
```powershell
Get-ChildItem -Path C:\,D:\ -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match 'identity_privilege|evidence_package|manuscript|submission_kit' } |
  Select-Object FullName, Length | Format-Table -AutoSize
```
찾으면 크기가 위 표와 일치하는지 먼저 보고, 최종적으로 SHA-256으로 확인:
```powershell
Get-FileHash <파일> -Algorithm SHA256
```

## 채우는 절차 (원본 확보 후)

논문·evidence 같은 대용량 zip은 git이 아니라 **GitHub Release**로 올리고,
이 문서에 Release 링크를 추가한다. 코드·프로토콜 등 경량 재현물만
`experiments/v1_v15/`(신설)에 넣는다. 동결 원칙상 내용 수정은 금지.
