"""
scripts/export_ca_bundle.py — 사내 프록시 환경용 CA 인증서 묶음 만들기 (Windows)

회사망 프록시가 HTTPS 를 중간에서 다시 서명하면(자체 서명 루트 CA), Python 의 기본 인증서
목록(certifi)에는 그 CA 가 없어 "CERTIFICATE_VERIFY_FAILED" 가 난다. 반면 Windows 인증서
저장소에는 회사 CA 가 설치돼 있으므로, 둘을 합친 PEM 파일을 만들어 넘기면
**인증서 검증을 끄지 않고** 통과할 수 있다(검증 비활성화보다 안전).

사용:
    python scripts/export_ca_bundle.py            # → storage/ca_bundle.pem
그 뒤 Pinecone 은 자동으로 이 파일을 사용(config.ca_bundle), arXiv 수집은 --ca-bundle 로 지정.
"""
from __future__ import annotations

import ssl
import sys
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "storage" / "ca_bundle.pem"


def build(out: Path = DEFAULT_OUT) -> int:
    import certifi

    if not hasattr(ssl, "enum_certificates"):
        sys.exit("Windows 전용입니다(ssl.enum_certificates 없음). 다른 OS 는 회사 CA 파일을 직접 지정하세요.")
    pems = [Path(certifi.where()).read_text(encoding="ascii")]
    n = 0
    for store in ("ROOT", "CA"):
        for der, enc, _trust in ssl.enum_certificates(store):
            if enc == "x509_asn":
                pems.append(ssl.DER_cert_to_PEM_cert(der))
                n += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(pems), encoding="ascii", errors="ignore")
    return n


if __name__ == "__main__":
    count = build()
    print(f"완료: certifi + Windows 인증서 {count}개 → {DEFAULT_OUT}")
