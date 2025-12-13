import os
import time
import json
import asyncio
import argparse
import aiohttp
import aiofiles
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from tqdm.asyncio import tqdm
from PIL import Image
import io

# =========================
# 설정 (Config)
# =========================
DEFAULT_TOP_N_ARTISTS = 2000
DEFAULT_IMAGES_PER_ARTIST = 100
DEFAULT_OUTPUT_DIR = "dataset"
DEFAULT_CONCURRENT_DOWNLOADS = 50  # 동시에 다운로드할 이미지 개수 (속도 조절 핵심)
DEFAULT_CONCURRENT_API_CALLS = 10  # API 동시 호출 제한

# Gelbooru API 설정
GELBOORU_API_KEY = os.getenv('GELBOORU_API_KEY', '')  # 환경변수 또는 빈 문자열
GELBOORU_USER_ID = os.getenv('GELBOORU_USER_ID', '')  # 환경변수 또는 빈 문자열
GELBOORU_API_URL = "https://gelbooru.com/index.php"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def parse_args():
    p = argparse.ArgumentParser(description="Danbooru artist list + Gelbooru image crawler")
    p.add_argument("--top-n-artists", type=int, default=DEFAULT_TOP_N_ARTISTS)
    p.add_argument("--images-per-artist", type=int, default=DEFAULT_IMAGES_PER_ARTIST)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--concurrent-downloads", type=int, default=DEFAULT_CONCURRENT_DOWNLOADS)
    p.add_argument("--concurrent-api-calls", type=int, default=DEFAULT_CONCURRENT_API_CALLS)
    p.add_argument(
        "--use-cache",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Whether to use cached artist list if present (non-interactive).",
    )
    p.add_argument(
        "--headless/--no-headless",
        dest="headless",
        default=True,
        help="Run Selenium in headless mode (default: enabled).",
    )
    p.add_argument("--gelbooru-api-key", type=str, default=None, help="Override GELBOORU_API_KEY env var")
    p.add_argument("--gelbooru-user-id", type=str, default=None, help="Override GELBOORU_USER_ID env var")
    return p.parse_args()

# =========================
# 1단계: Selenium으로 작가 목록 수집 (동기 방식)
# =========================
def fetch_artists_selenium(limit=DEFAULT_TOP_N_ARTISTS, headless=True):
    print(f"🚀 [Phase 1] Selenium으로 Danbooru 작가 {limit}명 수집 시작...")
    
    # Headless 설정 (속도 향상)
    chrome_options = Options()
    if headless:
        chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument(f'user-agent={HEADERS["User-Agent"]}')
    # 페이지 로딩 전략: eager (이미지 로딩 안 기다림 -> 속도 향상)
    chrome_options.page_load_strategy = 'eager' 

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    artists = []
    page = 1
    
    try:
        # 진행률 표시바 (tqdm)
        pbar = tqdm(total=limit, desc="Collecting Artists", unit="artist")
        
        while len(artists) < limit:
            url = f"https://danbooru.donmai.us/artists?commit=Search&page={page}&search%5Border%5D=post_count"
            driver.get(url)
            
            # 페이지 요소 찾기
            artist_elements = driver.find_elements(By.CSS_SELECTOR, "a.tag-type-1")
            
            if not artist_elements:
                print("\n더 이상 작가가 없습니다.")
                break
            
            new_count = 0
            for elem in artist_elements:
                if len(artists) >= limit:
                    break
                
                name = elem.text.strip()
                # 중복 및 금지된 작가 필터링
                if name and name not in artists and 'banned' not in name.lower():
                    artists.append(name)
                    new_count += 1
                    pbar.update(1)
            
            if new_count == 0:
                # 페이지는 로드됐으나 유효한 새 작가가 없으면 종료 (무한루프 방지)
                # 단, 페이지가 넘어가는 중일 수 있으므로 몇 번 더 시도하거나 종료 조건 정교화 필요
                # 여기서는 간단히 페이지 증가
                pass

            page += 1
            # Selenium은 너무 빠르면 차단될 수 있으니 최소한의 딜레이
            time.sleep(0.5)

    except Exception as e:
        print(f"\n❌ Selenium Error: {e}")
    finally:
        driver.quit()
        pbar.close()
    
    print(f"✓ 총 {len(artists)}명의 작가 목록 확보 완료")
    return artists

# =========================
# 2단계: 비동기 URL 수집 및 다운로드 (Asyncio)
# =========================

async def fetch_gelbooru_urls(session, artist, sem, images_per_artist: int, gelbooru_api_key: str, gelbooru_user_id: str):
    """
    Gelbooru API에서 비동기로 이미지 URL 수집
    - 개선점: 태그 공백 처리, 재시도 로직, 에러 상세 출력
    """
    # 1. 태그 정규화 (중요: Danbooru 'Name Name' -> Gelbooru 'Name_Name')
    # 공백을 언더스코어로 변경하고, 특수문자가 URL을 깨지 않도록 처리
    artist_tag = artist.strip().replace(' ', '_')
    
    image_urls = []
    
    # 2. 재시도 설정
    max_retries = 3
    
    async with sem: # API 동시 호출 제한
        pid = 0
        while len(image_urls) < images_per_artist:
            params = {
                'page': 'dapi',
                's': 'post',
                'q': 'index',
                'json': '1',
                'tags': artist_tag,  # 정규화된 태그 사용
                'limit': 100,
                'pid': pid
            }
            if gelbooru_api_key:
                params.update({'api_key': gelbooru_api_key, 'user_id': gelbooru_user_id})

            # 재시도 루프
            success = False
            for attempt in range(max_retries):
                try:
                    async with session.get(GELBOORU_API_URL, params=params, timeout=15) as response:
                        if response.status == 200:
                            # JSON 파싱 시도
                            try:
                                data = await response.json(content_type=None) # content_type 무시 (가끔 text/html로 옴)
                            except Exception as json_err:
                                # JSON 파싱 실패 시 (빈 결과 등) 중단하지 않고 로그 남김
                                # print(f"⚠️ [{artist}] JSON Parse Error on pid {pid}: {json_err}")
                                break # 이 페이지는 망가졌으므로 루프 탈출 (다음 pid로 가거나 종료)
                            
                            # 데이터 구조 확인
                            post_list = data.get('post', []) if isinstance(data, dict) else data
                            
                            if not post_list:
                                # 더 이상 이미지가 없으면 완전 종료
                                success = True 
                                break 
                            
                            # URL 추출
                            found_in_page = 0
                            for post in post_list:
                                if len(image_urls) >= images_per_artist:
                                    break
                                file_url = post.get('file_url') or post.get('image')
                                if file_url:
                                    image_urls.append(file_url)
                                    found_in_page += 1
                            
                            if found_in_page == 0:
                                # 데이터는 왔는데 URL이 없는 경우 (드묾)
                                break

                            success = True
                            break # 재시도 루프 탈출 (성공)
                        
                        elif response.status in [429, 500, 502, 503, 504]:
                            # 서버 부하/차단 시 잠시 대기 후 재시도
                            await asyncio.sleep(2 * (attempt + 1))
                            continue
                        else:
                            # 404 등 명확한 에러면 중단
                            print(f"❌ [{artist}] HTTP Error {response.status}")
                            break
                            
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    # 네트워크 에러 시 재시도
                    # print(f"⚠️ [{artist}] Network Error (Attempt {attempt+1}): {e}")
                    await asyncio.sleep(1)
                    continue
                except Exception as e:
                    print(f"❌ [{artist}] Unexpected Error: {e}")
                    break
            
            # 재시도 해도 실패했거나, 포스트가 비어서 success=True로 탈출한 경우 처리
            if not success or (success and not post_list):
                break
                
            pid += 1
            
            # 서버 친화적 딜레이 (너무 빠르면 차단됨)
            await asyncio.sleep(0.2)

    # 디버깅: URL을 하나도 못 찾은 경우 로그 출력
    if not image_urls:
        pass
        # print(f"  Result: [{artist}] -> Found 0 images (Tag used: {artist_tag})")
                
    return artist, image_urls

async def download_image(session, url, save_path, sem):
    """실제 이미지 다운로드 및 검증"""
    if os.path.exists(save_path):
        return True # 이미 존재하면 성공 처리

    async with sem: # 다운로드 동시 실행 제한
        try:
            async with session.get(url, timeout=45) as response:
                if response.status != 200:
                    return False
                content = await response.read()
                
                # 이미지 검증 (손상된 파일 방지)
                try:
                    img = Image.open(io.BytesIO(content))
                    img.verify() # 헤더 손상 확인
                    if img.width < 50 or img.height < 50: # 너무 작은 이미지 제외
                        return False
                except:
                    return False

                # 비동기 파일 쓰기
                async with aiofiles.open(save_path, 'wb') as f:
                    await f.write(content)
                return True
        except:
            return False

async def main_async_pipeline(
    artists,
    *,
    images_per_artist: int,
    output_dir: str,
    concurrent_downloads: int,
    concurrent_api_calls: int,
    gelbooru_api_key: str,
    gelbooru_user_id: str,
):
    """비동기 파이프라인 메인 로직"""
    print(f"\n🚀 [Phase 2] Gelbooru 이미지 수집 및 다운로드 시작 (Async)...")
    
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        # --- A. 모든 작가의 URL 먼저 수집 (메타데이터 확보) ---
        print("1. 작가별 이미지 주소(URL) 수집 중...")
        api_sem = asyncio.Semaphore(concurrent_api_calls)
        
        # 작가별 URL 수집 태스크 생성
        tasks = [
            fetch_gelbooru_urls(
                session,
                artist,
                api_sem,
                images_per_artist=images_per_artist,
                gelbooru_api_key=gelbooru_api_key,
                gelbooru_user_id=gelbooru_user_id,
            )
            for artist in artists
        ]
        
        all_download_targets = []
        
        # 완료되는 대로 큐에 넣기 (tqdm으로 진행상황 표시)
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching Metadata"):
            artist, urls = await f
            if not urls:
                continue
            
            # 폴더 생성
            artist_dir = os.path.join(output_dir, artist)
            os.makedirs(artist_dir, exist_ok=True)
            
            # 다운로드할 타겟 리스트 생성
            for idx, url in enumerate(urls):
                ext = url.split('.')[-1].split('?')[0][:4]
                # 파일명: 0.jpg, 1.png ...
                save_path = os.path.join(artist_dir, f"{idx}.{ext}")
                all_download_targets.append((url, save_path))

        print(f"\n✓ 총 {len(all_download_targets)}개의 다운로드 대상 확보.")

        # --- B. 대규모 병렬 다운로드 실행 ---
        print("2. 이미지 고속 다운로드 시작...")
        download_sem = asyncio.Semaphore(concurrent_downloads)
        
        # 다운로드 태스크 생성
        download_futures = [
            download_image(session, url, path, download_sem) 
            for url, path in all_download_targets
        ]
        
        success_count = 0
        # as_completed를 사용하여 완료되는 순서대로 진행바 업데이트
        for f in tqdm(asyncio.as_completed(download_futures), total=len(download_futures), desc="Downloading Images", unit="img"):
            if await f:
                success_count += 1
                
        print(f"\n🎉 모든 작업 완료! 성공적으로 다운로드된 이미지: {success_count}장")

# =========================
# 메인 실행부
# =========================
def main_cli() -> None:
    args = parse_args()

    # 윈도우 환경 asyncio 정책 설정
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    artist_list_path = os.path.join(output_dir, "top_artists.json")

    gelbooru_api_key = args.gelbooru_api_key if args.gelbooru_api_key is not None else GELBOORU_API_KEY
    gelbooru_user_id = args.gelbooru_user_id if args.gelbooru_user_id is not None else GELBOORU_USER_ID

    # 1) artists list
    artists = []
    cache_exists = os.path.exists(artist_list_path)
    use_cache = (args.use_cache in ("auto", "yes")) and cache_exists

    if use_cache:
        with open(artist_list_path, "r", encoding="utf-8") as f:
            artists = json.load(f).get("top_artists", [])
            if artists and isinstance(artists[0], dict):
                artists = [a["name"] for a in artists]
    else:
        artists = fetch_artists_selenium(limit=args.top_n_artists, headless=args.headless)

    # 목록 저장 (백업용)
    if artists:
        with open(artist_list_path, "w", encoding="utf-8") as f:
            json.dump({"top_artists": artists}, f, ensure_ascii=False, indent=2)

    # 2) download
    if artists:
        asyncio.run(
            main_async_pipeline(
                artists,
                images_per_artist=args.images_per_artist,
                output_dir=output_dir,
                concurrent_downloads=args.concurrent_downloads,
                concurrent_api_calls=args.concurrent_api_calls,
                gelbooru_api_key=gelbooru_api_key,
                gelbooru_user_id=gelbooru_user_id,
            )
        )
    else:
        print("작가를 찾지 못해 종료합니다.")


if __name__ == "__main__":
    main_cli()