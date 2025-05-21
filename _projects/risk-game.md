---
visible: true
layout: page
title: "Strategic Dominion: AI-Powered Risk Conquest"
authors: Lars C.P.M. Quaedvlieg, Ramon Reszat, Joan Botzev, Daniel Roder, Michael Balzer, Sree Kotala
description: Strategic Dominion is a project where we used the power of machine learning to develop agents for the classic game of Risk, enabling intelligent AI bots to engage in strategic conquests.
img: assets/img/risk.png
importance: 100
category: Research
developed_date: 2021-01-22 16:00:00-0000
---

This is group project was completed in Fall 2021 for the Project 2-1 course at my bachelor's [Data Science and Artificial Intelligence](https://www.maastrichtuniversity.nl/education/bachelor/data-science-and-artificial-intelligence/courses-curriculum)
at Maastricht University. 

## Phase 1: Developing the Game

In the first phase of the project, we spent our time developing the game in Java and laying out the backbone for the game-playing
agents. We utilized many useful design patterns to make sure the game mechanics, the bot, and the user-interface were mainly
disconnected and could be developed simultaneously.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <div class="text-center">
            {% include figure.liquid path="assets/img/phase1-risk.gif" title="Example of the solution" class="center img-fluid rounded z-depth-1" %}
        </div>
    </div>
</div>
<div class="caption">
    This GIF shows an example of the product at the end of this phase.
</div>

## Phase 2 and 3:

The remaining two phases were spent both on an improved user-interface and the development of two bot players. For the bots,
we used Double Deep Q-Learning in order to estimate the Q-values of certain actions. Since the game is very stochastic, partially observable, and multi-agent, some assumptions
had to be made in order to make developing a bot feasible. Instead of being concerned with global actions, the agent 
learned, for example, the number of troops to choose for an attack. This is still useful for the bot while playing the game, but
simplifies the state and action spaces dramatically.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <div class="text-center">
            {% include figure.liquid path="assets/img/phase2-risk.gif" title="Example of the solution" class="center img-fluid rounded z-depth-1" %}
        </div>
    </div>
</div>
<div class="caption">
    This GIF shows an example of the final product.
</div>

If you are interested in the details of the project, the final report for this project is attached below!

{% pdf "/assets/pdf/dke_project21.pdf" height=1030px no_link %}