---
layout: page
permalink: /publications/
title: Publications
description: Publications by categories in reverse chronological order (* = equal contributions). 
nav: true
nav_order: 1
---

<!-- _pages/publications.md -->

<!-- Bibsearch Feature -->
<!-- {% include bib_search.liquid %} -->

<div class="publications">
    {% bibliography --query @*[year>=2022] %}
</div>
