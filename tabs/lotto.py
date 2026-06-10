import streamlit as st
import os, json, random
from datetime import datetime


@st.cache_data
def load_lotto_history():
    if os.path.exists("lotto_data.json"):
        with open("lotto_data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def render():
    st.header("🍀 로또 번호 생성기")
    st.markdown("행운의 로또 번호 6자리를 생성해보세요!")

    past_draws = load_lotto_history()

    if "lotto_history" not in st.session_state:
        st.session_state["lotto_history"] = []

    if st.button("번호 생성", key="lotto_btn"):
        lotto_numbers = sorted(random.sample(range(1, 46), 6))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        history_stats = {1: 0, 3: 0, 4: 0, 5: 0}
        for draw_no, numbers in past_draws.items():
            match_count = len(set(lotto_numbers) & set(numbers))
            if match_count == 6:
                history_stats[1] += 1
            elif match_count == 5:
                history_stats[3] += 1
            elif match_count == 4:
                history_stats[4] += 1
            elif match_count == 3:
                history_stats[5] += 1

        total_wins = sum(history_stats.values())
        st.session_state["lotto_history"].insert(0, {
            "time": now,
            "numbers": lotto_numbers,
            "history_stats": history_stats,
            "total_wins": total_wins,
        })

    for idx, record in enumerate(st.session_state["lotto_history"]):
        total_wins = record.get("total_wins", 0)
        if total_wins > 0:
            stats = record["history_stats"]
            summary = []
            if stats[1] > 0: summary.append(f"1등 {stats[1]}회")
            if stats[3] > 0: summary.append(f"3등(또는 2등) {stats[3]}회")
            if stats[4] > 0: summary.append(f"4등 {stats[4]}회")
            if stats[5] > 0: summary.append(f"5등 {stats[5]}회")
            st.markdown(f"🕒 **생성 일시:** {record['time']} | 🎯 **역대 당첨 이력 총 {total_wins}회!** ({', '.join(summary)})")
        else:
            st.caption(f"🕒 생성 일시: {record['time']} | 😢 역대 당첨 내역 없음")

        html_content = "<div style='display: flex; gap: 15px; margin-top: 5px; margin-bottom: 10px;'>"
        for num in record["numbers"]:
            if num <= 10: color = "#fbc400"
            elif num <= 20: color = "#69c8f2"
            elif num <= 30: color = "#ff7272"
            elif num <= 40: color = "#aaaaaa"
            else: color = "#b0d840"
            html_content += f"<div style='width: 50px; height: 50px; border-radius: 50%; background-color: {color}; color: white; display: flex; justify-content: center; align-items: center; font-size: 20px; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); box-shadow: 2px 2px 5px rgba(0,0,0,0.2);'>{num}</div>"
        html_content += "</div>"

        st.markdown(html_content, unsafe_allow_html=True)

        if idx < len(st.session_state["lotto_history"]) - 1:
            st.divider()
