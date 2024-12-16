# DatingRL
Reinforcement learning solves the dating problem?

## Motivation

> What would you do if faced with "a choice that is **immediate** and **final**?

Imagine you are driving down a straight road and need to fill up on gas soon. Each gas station has some posted price (we'll assume signages are honest) and you, as a rational person, would like to buy the cheapest gas. What would you do?

This is the secretary problem in a nutshell.  
https://en.wikipedia.org/wiki/Secretary_problem

Thinking through this problem iteratively:

* You would probably not fill up at the first gas station you encounter (unless the price was something absurdly low or even free?!).
* For every additional gas station you visit you get some additional information to assess _future_ gas stations but you lose out if the current gas station is the cheapest. (Assume that you're on a road trip and that you're not going to backtrack 50 miles when you realize that gas was cheaper somewhere behind you.)
* Therefore, you need to choose some condition for when to stop collecting data and start selecting for the cheapest gas.
* For the secretary problem, the optimal strategy follows this structure where you first have to learn "what's out there" and then start looking for the best you've seen so far.
* I hypothesize that, even when the assumptions are slightly different from the classical secretary problem, in general, for scenarios where you cannot aggregate all choices and then decide, the optimal strategy will also follow this pattern.

As it turns out if you do the math, the optimal strategy for the secretary problem is to unconditionally reject n/e (where e is Euler's constant) gas stations and then pick the next cheapest gas station. (If you run out of gas or candidates then you just have to go with the last one.)

This strategy will find you the cheapeast gas station with probability 1/e. In other words, 37% of the times you will get the cheapest gas.

Veritasium has a great video on this:  
https://youtu.be/d6iQrh2TK98?si=nSDDEYlWVr-HLh71&t=724

You might now be asking:  
**Wait!!? What does finding the cheapeast gas station have anything to do with my dating life?????**

Imagine you have the patience to date 1000 people or be available for 10 years. Assuming that rejecting anyone is final, the secretary problem models this problem pretty well (when the number of candidates, "n" is known -- also works for time but I think the proof is technically different).

Actually that's a lie, there's one big thing the secretary problem assumes and overlooks.

> "You can reject a gas station but a gas station will never reject you." - Unknown

Ok yes I made up that quote. There's nothing in the secretary problem that encodes the real world situation where a candidate turns down a job offer (when you are the interviewer) or your date rejects you!

This motivation section is getting long. If you couldn't guess by now that's basically why I made this project.  
**This repository investigates how violating the secretary problem's assumptions and modeling these kinds of "best choice problems" more realistically affects the optimal solution.**
