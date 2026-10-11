// Live page: poll /api/status every second and update the state panel.
(function () {
  const $ = (id) => document.getElementById(id);
  const stateEl = $("state"), alertEl = $("alert"), armBtn = $("arm");
  let armed = true;

  armBtn.addEventListener("click", async () => {
    armBtn.disabled = true;
    try {
      await fetch("/api/arm", { method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ armed: !armed }) });
      await refresh();
    } finally { armBtn.disabled = false; }
  });

  function show(alertClass, text) {
    alertEl.className = "alert " + alertClass;
    alertEl.textContent = text;
  }

  async function refresh() {
    try {
      const r = await fetch("/api/status", { cache: "no-store" });
      const s = await r.json();
      if (!s.monitor_running || !s.state) {
        stateEl.className = "state state-unknown";
        stateEl.textContent = s.monitor_running ? "Starting…" : "Monitor stopped";
      } else {
        stateEl.className = "state state-" + s.state.split(" ")[0];
        stateEl.textContent = s.state;
      }
      armed = s.armed !== false;
      armBtn.hidden = !s.monitor_running;
      armBtn.textContent = armed ? "Pause monitoring" : "Resume monitoring";
      armBtn.className = "button arm-button " + (armed ? "secondary" : "");
      $("source").textContent = s.source || "–";
      $("movement").textContent = s.movement == null ? "off" :
        `${s.movement.toFixed(3)} (alert level ${s.movement_threshold})`;
      $("rhythm").textContent = s.rhythm_s != null ? `${s.rhythm_s.toFixed(1)} s / ${s.onset_s.toFixed(0)} s needed` : "–";
      $("body").textContent = s.body_visible == null ? "–" : (s.body_visible ? "Yes" : "No");

      if (s.call) show("danger", "No response yet: escalated. Phone call flag is set.");
      else if (s.acknowledged_by) show("ok", `${s.acknowledged_by} is with them.`);
      else if (s.incident_active) show("warn", "Alert sent. Waiting for someone to respond.");
      else if (s.alarm_active) show("warn", "Local alarm is sounding.");
      else alertEl.className = "alert hidden";
    } catch (e) {
      stateEl.className = "state state-unknown";
      stateEl.textContent = "Connection lost";
    }
  }
  refresh();
  setInterval(refresh, 1000);
})();
