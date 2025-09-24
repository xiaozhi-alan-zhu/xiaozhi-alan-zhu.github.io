---
layout: page
title: "Home"
permalink: /
---

<div class="home-grid">
  <aside class="sidebar">
    <div class="avatar-wrap">
      <img class="avatar" src="{{ site.author.avatar | default: '/assets/images/photo.png' | relative_url }}" alt="Profile photo">
    </div>
    <h2 class="author-name">{{ site.author.name }}</h2>
    {% if site.author.affiliation %}<div class="affiliation">{{ site.author.affiliation }}</div>{% endif %}

    <ul class="contact">
      {% if site.author.location %}<li>📍 {{ site.author.location }}</li>{% endif %}
      {% if site.author.email %}<li>✉️ <a href="mailto:{{ site.author.email }}">{{ site.author.email }}</a></li>{% endif %}
      {% if site.author.scholar %}<li>🎓 <a href="{{ site.author.scholar }}" target="_blank" rel="noopener">Google Scholar</a></li>{% endif %}
      {% if site.author.github %}<li>🐙 <a href="{{ site.author.github }}" target="_blank" rel="noopener">GitHub</a></li>{% endif %}
    </ul>
  </aside>

  <section class="main">
    <h2>Biography</h2>
    <p>
      My name is Xiaozhi(Alan) Zhu and I’m currently a research scientist at Meta Platform, inc. 
      Before joining Meta, I obtained my Ph.D. in <a href="https://acms.nd.edu/" traget="_blank" rel="noopener">applied math from University of Notre Dame</a> with 
      <a href="https://www3.nd.edu/~yzhang10/" target="_blank" rel="noopener">Prof. Yong-tao Zhang</a>
      being my advisor.
      My research interests focusing on numerical PDE, ML for physics simulation and related topics.
    </p>

    <hr class="section-divider">

    <h2>News</h2>
    <ul class="news-list">
      {% for item in site.data.news %}
        <li><span class="news-date">[{{ item.date }}]</span> {{ item.text }}</li>
      {% endfor %}
    </ul>
  </section>
</div>