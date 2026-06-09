import asyncio
import aiohttp
import json

async def fetch_draw(session, draw_no):
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}"
    try:
        async with session.get(url) as response:
            data = await response.json()
            if data.get("returnValue") == "success":
                numbers = [data[f"drwtNo{i}"] for i in range(1, 7)]
                return str(draw_no), numbers
    except:
        pass
    return None

async def main():
    results = {}
    async with aiohttp.ClientSession() as session:
        # Assuming around 1200 draws so far (as of mid 2026 it's ~1220)
        tasks = [fetch_draw(session, i) for i in range(1, 1250)]
        for future in asyncio.as_completed(tasks):
            res = await future
            if res:
                results[res[0]] = res[1]
                
    with open("lotto_data.json", "w", encoding="utf-8") as f:
        json.dump(results, f)

if __name__ == "__main__":
    asyncio.run(main())
