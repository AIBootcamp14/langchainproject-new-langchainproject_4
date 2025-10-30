"""유틸리티 함수 모음
공통으로 사용되는 헬퍼 함수들"""

import os
import json
import h as hlib
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging
from pathlib import Path


# 로거 설정
logger=logging.getLogger(__name__)


#|(유틸리티
def ensure_directory(path: str ) -> Path:
"""
디렉토리가존재하지않으면생성
"""
dir_path=Path(path)
dir_path.mkdir(parents=True, exist_ok=True)
return dir_path


def read_json_file(file_path: str ) -> Dict[str, Any]:
"""
JSON 파일 읽기
"""
try:
with open(f:
return json.load(f)
except FileNotFoundError:
logger.warning(f"|D>DÆµÈä:{file_path}")
return {}
except json.JSONDecodeError as e:
logger.error(f"JSONñ$X:{e}")
return {}


def write_json_file(file_path: str, data: Dict[str, Any], indent: int=2):
"""
JSON 파일 쓰기
"""
try:
with open(f:
json.dump(data, f, ensure_ as cii=False, indent=indent)
logger.info(f"파일 저장 완료:{file_path}")
except Exception as e:
logger.error(f"파일 저장 실패:{e}")


#M¤¸처리유틸리티
def clean_text(text: str) -> str:
"""
텍스트정제(공백정리, 특수문자제거)
"""
#ðõ1DX\
text=re.sub(r'\s+','', text)

#^¤õ1p
text=text.strip()

#HTMLÜøp
text=re.sub(r'<[^>]+>','', text)

#¹8처리
text=text.replace('\u200b',''#Zero-widthspace
text=text.replace('\ufeff',''#BOM

return text


def truncate_text(text: str, max_length: int=100, suffix: str="...") -> str:
"""
텍스트를지정된길이로자르기
"""
if len(text <= max_length:
return text

return text[: max_length-len(suffix)]+suffix


def extract_keywords(text: str, top_k: int=5 ) -> List[str]:
"""
텍스트에서키워드추출(간단한버전)
"""
#8타입X¹8p
text_lower=text.lower()
words=re.findall(r'\b[a-z]+\b', text_lower)

#사용´p(간단한목록)
stopwords={
'the','a','an','is','are','was','were','be','been',
'have','has','had','do','does','did','will','would',
'could','should','may','might','must','can','this',
'that','these','those','i','you','he','she','it',
'we','they','what','which','who','when','where',
'why','how','all','each','every','both','few',
'more','most','other','some','such','no','nor',
'not','only','own','same','so','than','too',
'very','just','in','on','at','to','for','of',
'with','by','from','up','out','if','about','into',
'through','during','before','after','above','below',
'between','under','again','further','then','once'
}

filtered_words=[wfor w in wordsifwnot in stopwordsandlen(w) >2]

#단어빈도계산
word_freq={}
for wordinfiltered_words:
word_freq[word]=word_freq.get(word, 0) + 1

#ÁkX
sorted_words=sorted(word_freq.items(), key=lambdax: x[1], reverse=True)
return [wordfor word , freqinsorted_words[: top_k]]


#tÜID선택1
def generate_h as h(text: str) -> str:
"""
텍스트의SHA256해시생성
"""
return h as hlib.sha256(text.encode('utf-8')).hexdigest()


def generate_document_id(content: str, metadata: Optional[Dict]=None) -> str:
"""
문서ID생성
"""
#내용üT타입데이터|°iXìID 생성
id_source=content

if metadata:
#UR테이블사용t<tìh
if'url'in metadata:
id_source=f"{metadata['url']}_{content[:100]}"
elif'title'in metadata:
id_source=f"{metadata['title']}_{content[:100]}"

return generate_h as h(id_source)[:16]#16처리Ì사용자


#Ü(유틸리티
def get_timestamp() -> str:
"""
현재타임스탬프반환
"""
return datetime.now().isoformat()


def format_timestamp(timestamp: Union[str, datetime], format_str: str="%Y-%m-%d%H:%M:%S") -> str:
"""
타임스탬프포맷팅
"""
if isinstance(timestamp, str):
dt=datetime.fromisoformat(timestamp)
else:
dt=timestamp

return dt.strftime(format_str)


def time_ago(timestamp: Union[str, datetime]) -> str:
"""
ÁÜ로드(:"3Ü")

Args:
timestamp:타입¤ì

Returns:
ÁÜ8컴포넌트
"""
if isinstance(timestamp, str):
dt=datetime.fromisoformat(timestamp)
else:
dt=timestamp

now=datetime.now()
diff=now-dt

seconds=diff.total_seconds()
if seconds<60:
return ")"
elif seconds<3600:
minutes=int(seconds/60)
return f"{minutes}"
elif seconds<86400:
hours=int(seconds/3600)
return f"{hours}Ü"
elif seconds<2592000:
days=int(seconds/86400)
return f"{days}|"
else:
months=int(seconds/2592000)
return f"{months}Ô"


#환경 변수(
def get_env_variable(key: str, default: Optional[str]=None, required: bool=False ) -> Optional[str]:
"""
환경변수가져오기
"""
value=os.getenv(key, default)

if required and value is None:
raiseValueError(f"D환경 변수$타입JXµÈä:{key}")

return value


def load_env_file(env_path: str=".env"):
"""
.env파일 전로드
"""
try:
from dotenv import load_dotenv
load_dotenv(env_path)
logger.info(f"환경 변수|로드:{env_path}")
except ImportError:
logger.warning("python-dotenv$X타입JXµÈä.")
except Exception as e:
logger.error(f"환경 변수|로드실패:{e}")


#유틸리티
def is_valid_url(url: str) -> bool:
"""
URL유효성검사
"""
url_pattern=re.compile(
r'^https?://'#http://or https://
r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0, 61}[A-Z0-9])?\.)+[A-Z]{2, 6}\.?|'#domain...
r'localhost|'#localhost...
r'\d{1, 3}\.\d{1, 3}\.\d{1, 3}\.\d{1, 3})'#...or ip
r'(?::\d+)?'#optionalport
r'(?:/?|[/?]\S+)$', re.IGNORECASE
)
return url_pattern.match(url is not None


def is_valid_email(email: str) -> bool:
"""
이메일 전유효성검사
"""
email_pattern=re.compile(
r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)
return email_pattern.match(email is not None


#데이터타입X유틸리티
def safe_json_loads(json_str: str, default: Any=None ) -> Any:
"""
안전한JSON파싱
"""
try:
return json.loads(json_str)
except(json.JSONDecodeError,대화ypeError):
return default


def dict_to_readable_text(data: Dict[str, Any], indent: int=0) -> str:
"""
딕셔너리를읽기쉬운텍스트로변환
"""
lines=[]
indent_str="  " * indent

for key, value in data.items():
if isinstance(value, dict):
lines.append(f"{indent_str}{key}:")
lines.append(dict_to_readable_text(value, indent+1))
elif isinstance(value, list):
lines.append(f"{indent_str}{key}:")
for iteminvalue:
if isinstance(item, dict):
lines.append(dict_to_readable_text(item, indent+1))
else:
lines.append(f"{indent_str}-{item}")
else:
lines.append(f"{indent_str}{key}:{value}")

return "\n".join(lines)


#배치처리유틸리티
def batch_process(items: List[Any], batch_size: int=10 ) -> List[List[Any]]:
"""
리스트를배치로나누기
"""
batches=[]
foriinrange(0, len(items), batch_size):
batches.append(items[i:i+batch_size])
return batches


#성능 측정유틸리티
cl as s대화imer:
"""Ü!èM¤¸äÈ"""

def __init__(self, name: str="Operation"):
self.name=name
self.start_time=None
self.elapsed_time=None

def __enter__(self):
self.start_time=datetime.now()
return self

def __exit__(self, exc_type, exc_val, exc_tb):
self.elapsed_time=(datetime.now() -self.start_time).total_seconds()
logger.info(f"{self.name}Ü:{self.elapsed_time:.2f}")

def get_elapsed_time(self) -> float:
"""½üÜX"""
if self.elapsed_timeisnotNone:
return self.elapsed_time
elif self.start_timeisnotNone:
return(datetime.now() -self.start_time).total_seconds()
return 0.0


#포맷 유틸리티
def format_bytes(bytes_size: int) -> str:
"""
바이트크기를읽기쉬운형식으로변환
"""
for unitin['B','KB','MB','GB','TB']:
if bytes_size<1024.0:
return f"{bytes_size:.1f}{unit}"
bytes_size/=1024.0
return f"{bytes_size:.1f}PB"


#에서ì처리유틸리티
def safe_execute(func,*args, default=None,**kwargs):
"""
안전한함수실행(예외처리)
"""
try:
return func(*args,**kwargs)
except Exception as e:
logger.error(f"hä$X:{e}")
return default


#테스트유틸리티
if __name__=="__main__":
print(">ê유틸리티h테스트\n")

#M¤¸처리테스트
print("1ãM¤¸처리")
test_text="Thisisatest<b>text</b>with spaces"
print(f"원본:'{test_text}'")
print(f"처리:'{clean_text(test_text)}'")
print(f"자르기:'{truncate_text(test_text, 20)}'")
print(f"키워드:{extract_keywords(test_text)}")

#해시 생성 테스트
print("\n2ã해시 생성")
test_content="Testdocumentcontent"
doc_id=generate_document_id(test_content,{'title':'TestDoc'})
print(f"8ID:{doc_id}")

#Ü(테스트
print("\n3ãÜ처리")
timestamp=get_timestamp()
print(f"현재 시간:{timestamp}")
print(f"ì÷Ü:{format_timestamp(timestamp)}")

#URL테스트
print("\n4ã유틸리티")
test_url="https://example.com/page"
test_email="user@example.com"
print(f"URL'{test_url}'유효:{is_valid_url(test_url)}")
print(f"Email'{test_email}'유효:{is_valid_email(test_email)}")

#배치처리테스트
print("\n5ã배치처리")
items=list(range(25))
batches=batch_process(items, batch_size=10)
print(f"25개를10배치:{len(batches)}배치")
for i, batch in enumerate(batches):
print(f"배치{i+1}:{len(batch)}m사용")

#성능 측정 테스트
print("\n6ã성능 측정")
import time
withTimer("테스트Å") as timer:
time.sleep(0.5)
print(f"!Ü:{timer.get_elapsed_time():.2f}")

print("\n유틸리티h테스트완료!")