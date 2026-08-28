# Amendment 03. NumPy superclass 문자열 직렬화 호환성

작성 시점: 19/19 participant neural caches가 모두 생성된 뒤, 첫 participant의 summary correlation 또는 gate를 집계하기 전

Cache의 `superclasses`가 Pandas에서 전달된 object dtype으로 저장되었다. 최종 집계 코드는 `allow_pickle=False`로 NPZ를 열었기 때문에 object string array 접근 시 중단되었다. Base/adapted 및 neural arrays는 수치 array로 정상 저장되었으며, 중단 시점에는 participant correlation이나 gate가 아직 계산되지 않았다.

수정은 본 분석이 직접 생성한 로컬 cache를 `allow_pickle=True`로 열고 `superclasses`를 즉시 string array로 변환하는 것으로 한정한다. 수치 arrays, 참가자, 시간창, 모델, 대역, endpoints 및 gates는 변경하지 않는다. 이후 새 cache를 만들 경우 superclass는 명시적 Unicode dtype으로 저장한다.
