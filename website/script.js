(function () {
  "use strict";
  var CHAT_URL = ((window.CEDARSUN_CONFIG || {}).chatUrl || "http://127.0.0.1:7860").replace(/\/$/, "");

  // mobile menu
  var toggle = document.querySelector(".nav-toggle");
  var nav = document.getElementById("nav");
  toggle.addEventListener("click", function () {
    var open = nav.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
  nav.addEventListener("click", function (e) {
    if (e.target.tagName === "A") { nav.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); }
  });

  // highlight the nav link of the section on screen
  var links = Array.prototype.slice.call(nav.querySelectorAll("a"));
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          links.forEach(function (a) { a.classList.toggle("active", a.getAttribute("href") === "#" + en.target.id); });
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    links.forEach(function (a) {
      var t = document.querySelector(a.getAttribute("href"));
      if (t && t.id) io.observe(t);
    });
  }

  document.getElementById("year").textContent = new Date().getFullYear();

  // ---- chat widget
  var fab = document.getElementById("chat-fab");
  var panel = document.getElementById("chat-panel");
  var frame = document.getElementById("chat-frame");
  var offline = document.getElementById("chat-offline");
  var loaded = false;

  function checkChat() {
    // The iframe cannot tell us if it failed, so ping the server first.
    var done = false;
    var timer = setTimeout(function () { if (!done) { done = true; offline.hidden = false; } }, 4000);
    fetch(CHAT_URL + "/config", { mode: "no-cors" }).then(function () {
      if (done) return; done = true; clearTimeout(timer); offline.hidden = true;
    }).catch(function () {
      if (done) return; done = true; clearTimeout(timer); offline.hidden = false;
    });
  }

  function openChat() {
    panel.hidden = false;
    fab.setAttribute("aria-expanded", "true");
    if (!loaded) { frame.src = CHAT_URL; loaded = true; checkChat(); }
  }
  function closeChat() { panel.hidden = true; fab.setAttribute("aria-expanded", "false"); }

  fab.addEventListener("click", function () { panel.hidden ? openChat() : closeChat(); });
  document.getElementById("chat-close").addEventListener("click", closeChat);
  document.querySelectorAll("[data-open-chat]").forEach(function (b) { b.addEventListener("click", openChat); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && !panel.hidden) closeChat(); });

  // ---- contact form -> the assistant's lead API (saved exactly like leads taken in the chat)
  var form = document.getElementById("lead-form");
  var statusEl = document.getElementById("lead-status");
  function say(msg, cls) { statusEl.textContent = msg; statusEl.className = "form-note " + (cls || ""); }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var f = new FormData(form);
    var name = (f.get("name") || "").trim(), email = (f.get("email") || "").trim();
    if (!name) return say("Please enter your name.", "err");
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]{2,}$/.test(email)) return say("Please enter a valid email address.", "err");
    say("Sending...", "");
    var btn = form.querySelector("button[type=submit]"); btn.disabled = true;

    fetch(CHAT_URL + "/gradio_api/call/lead", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: [name, email, f.get("area") || "", f.get("message") || ""] })
    })
      .then(function (r) { if (!r.ok) throw new Error("bad status"); return r.json(); })
      .then(function (j) { return fetch(CHAT_URL + "/gradio_api/call/lead/" + j.event_id); })
      .then(function (r) { return r.text(); })
      .then(function (t) {
        if (t.indexOf("event: complete") === -1 || !/status\\?"\s*:\s*\\?"ok/.test(t)) throw new Error("not saved");
        say("Thank you, " + name.split(" ")[0] + "! A CedarSun specialist will contact you within 1 business day.", "ok");
        form.reset();
      })
      .catch(function () {
        say("We could not send your request right now. Please email hello@cedarsun-energy.example or use the chat.", "err");
      })
      .then(function () { btn.disabled = false; });
  });
})();
