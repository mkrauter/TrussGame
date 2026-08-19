# Can a machine learn engineering intuition?

*I built a small game about how structures bend, then spent a few weeks
teaching a computer to play it. It fooled me completely, and then sat on a shelf
for years before I found out why. What finally worked had nothing to do with a
bigger model — and quite a lot to do with admitting I had picked the wrong kind
of machine.*

---

Here is a puzzle. Ten steel joints, scattered at random and wired together into
triangles. The two outermost joints are bolted to the ground. Now hang a weight
on one of the others.

Where does that joint end up?

Not straight down — that is the first thing everyone gets wrong. A joint in a
truss cannot go wherever it likes. It is held by the members attached to it, and
they are held by their members, and so on out to the ground. The joint slides
along whatever path the whole structure permits, which is often sideways, and
occasionally *upwards*.

You cannot calculate this in your head. But you can develop a feel for it. Play
a few rounds and you start reading the shape — this one will swing left, this
one will barely move — without knowing why you know. Engineers call it intuition.
It is one of the more mysterious things a human brain does, and it is the reason
I wanted to find out whether a machine could do it too.

## The game

Years ago, when I joined a large structural engineering firm, I wanted to show
colleagues how much fun writing code can be, and how much you can do in very few
lines of it. So I wrote a small game: it draws a random truss, you click where
you think the loaded joint will settle, and then it runs the simulation and
scores you.

The whole thing — the physics, the graphics, the animation — came to
eighty-five lines.

The scoring matters for what follows, so bear with me for one paragraph.
**100%** means you clicked exactly where the joint settled. **0%** means you
missed by at least as far as the joint actually travelled. So your score is the
*share of the movement* you predicted, not a distance. Guess the joint's
starting position and you score exactly zero, however small the movement was.

Human players, after a few rounds of practice, land somewhere around 70–80%.

## The obvious idea

If a person can learn this by looking, could a neural network?

The natural fit seemed to be the kind of network that recognises objects in
photographs. Those work by scanning an image with small filters that detect
local patterns — an edge here, a corner there — and stacking those detections
into progressively larger ideas. It is the technology behind every "what breed
is this dog" demo you have ever seen, and it is very good.

So: feed it a screenshot of the truss, ask it for two numbers — the coordinates
where the joint will end up — and show it a few thousand examples with the
correct answers. Standard stuff. It trained in about ten minutes on a free
cloud GPU.

That was 2021, and the whole thing took a few weeks of evenings. It scored
around 60%, and I was delighted. It seemed to have picked up something real
about mechanics just by staring at pictures.

Then life happened, and the project went quietly to sleep for the next several
years.

## The question I forgot to ask

When I finally dug the project out again, I started with the one thing I should
have done on the first afternoon and never got around to: work out what the
*dumbest possible strategy* scores.

Not a clever comparison. The dumbest one. Ignore the truss entirely. Find the
loaded joint, and move it straight down by the average distance that joints
move. One line of arithmetic, no learning, no understanding.

**That scores 59.5%.**

My network scored 60.5%.

The thing I had been quietly proud of for years was worth one percentage point
over a rule you could write on a napkin.

Worse, when you pull the model apart, that is precisely what it was doing. It
located the loaded joint in the image almost perfectly — it was excellent at
*seeing*. But its guess about how far the joint would move was, measurably, worse
than just always guessing the average. It shifted the joint by about 135 pixels
when the truth averaged 155, no matter what the structure looked like.

It had never learned mechanics at all. It had learned to find the blue dot and
nudge it downward. And because the scoring rule is generous to anyone who knows
roughly how far things move, that was enough to look clever.

There is a line the physicist Enrico Fermi liked to quote from John von Neumann,
about theories that fit the data for the wrong reasons:

> *"With four parameters I can fit an elephant, and with five I can make him
> wiggle his trunk."*

My model had 307,618 of them. It had fitted a rather good elephant.

## Trying harder does not help

My first instinct was that the network was simply too crude, and it was — built
by a beginner, with several rookie mistakes in it. So I rebuilt it properly,
with everything the last few years of practice would recommend.

That got it to **77%**. Real progress, and past the napkin rule at last.

Then I found what looked like the smoking gun.

These networks have something called a *receptive field*: the amount of the
picture that any one part of the network can actually see at once. Because they
work by combining small local patches, that field is limited, and mine was about
half the width of a typical truss. **No part of the network had ever seen both
supports at the same time.** I was asking it about a structure it could not look
at in one go.

That is a textbook problem with a textbook fix. I widened the field until it
comfortably covered the whole truss, at no extra cost in model size, and
retrained, confident this was the moment.

It got **worse**. Not by much. But consistently — and the version that could see
furthest was the one that started off worst.

I stared at that result for a long time. It is the single most useful thing that
happened in this project, because it meant my diagnosis was wrong, and wrong in a
way that pointed somewhere specific.

## A truss is not a picture. It is an argument.

Here is what I had missed.

When you load a truss, every joint's final position depends on *every other
joint*. Pull one corner of a fishing net and the whole net rearranges itself.
There is no such thing as a local answer: the joint you care about cannot settle
until its neighbours settle, and they cannot settle until their neighbours do,
all the way to the ground and back.

Image networks are built on the opposite assumption. They work brilliantly for
pictures precisely because pictures *are* mostly local — a cat's ear is a cat's
ear regardless of what is happening in the far corner of the photo. Stacking
local detectors lets information travel a certain distance, and that distance is
what the receptive field measures.

But no amount of stacking local detectors turns them into something that
resolves an all-at-once, everything-depends-on-everything problem. Reach was
never what was missing. I had been trying to fix the size of the tool when the
issue was that I had brought the wrong tool.

## How engineers actually do it

The fix came from looking at how these problems are solved without any machine
learning at all.

For a structure of real size, nobody solves the equations in one shot. They
solve them *iteratively*: each joint looks at the joints it is physically
connected to, works out how hard they are pulling on it, adjusts itself
slightly, and then everyone does it again. Round after round, the whole
structure converges on its answer, and information travels outward through the
members exactly as force does.

Read that description again and it is not really an algorithm. It is a
conversation. Each joint talks to its neighbours, over and over, until they all
agree.

And that turns out to describe a completely different family of neural network —
one that operates on a *network of connected things* rather than on a grid of
pixels. Joints become nodes. Members become the channels they talk along. One
round of the conversation is one round of the network. Instead of asking a
picture-reader to somehow imply the physics, you build a machine whose shape is
already the shape of the problem.

Two more decisions made it work, and both are about restraint rather than
cleverness:

**Tell it what you already know.** I know exactly how stiff each member is —
that is simple geometry. So I hand that to the network as an input instead of
making it rediscover the concept from scratch. I know the two supports do not
move, so I fix them rather than hoping the network learns to hold them still.
Every one of those is capacity not wasted relearning something I could just write
down.

**Let physics mark its own homework.** I know the equation the true answer has to
satisfy — essentially, that all the forces at every joint must cancel out. So
during training I can check any guess against that equation directly, *without
needing to know the right answer*. The network gets told not just "you were
wrong" but "your answer is not in equilibrium", which is a far more useful
complaint.

## 42,498 parameters

That is the whole model. About a seventh the size of the one that was fitting
elephants — and it scores **96%**.

More telling than the score: as training goes on, its answers drift steadily
closer to satisfying the equilibrium equation. It is not memorising which
pictures go with which answers. It is converging on solutions that are
*physically consistent*, which is a fair description of what it means to
understand something.

## "Hang on — that's cheating"

Quite right, and this was the objection I had to take most seriously.

The founding rule of the project was that the machine gets exactly what a human
player gets: the picture on screen, nothing else. Handing it a tidy list of joint
coordinates is not playing the game.

So I measured how much that rule actually costs. Take a *perfect* solver and
blur its reading of where the joints are, the way an eye blurs things. At the
precision a person reads a screen — a pixel or three — a perfect solver still
scores 97–99%. Give it exact coordinates and it scores 100%.

In other words: **crisp eyesight is worth about two points. Doing the mechanics
is worth forty.** The seeing was never the hard part, which in hindsight is
exactly what the first model was telling me when it located the joint perfectly
and then guessed the average.

That is interesting, and it is still not a reason to break the rule. So the
current version reads the screen. It just does not use the same machinery for
looking and for thinking:

- A small network — 9,101 parameters, tiny — scans the image and finds the
  joints, working out which are supports and which is loaded.
- Which joints are connected is settled by simply *looking*: is there a line
  drawn between these two? That part is not learned at all, and should not be.
  "Is there a line here" is a question you answer by checking; a neural network
  could only add a way to get it wrong.
- The conversation network then does the mechanics on the structure that came
  out of that.

Picture-readers do what picture-readers are genuinely brilliant at, and the
structural reasoning happens somewhere better suited to it. End to end, from
pixels alone: **96.3%**. Almost all of the remaining error is eyesight — a
flawless solver with the same eyes would score about 98%.

## Difficulty you can measure in thinking

There is a small delight at the end of this.

Because the machine works by repeating the same conversation round after round,
you can simply ask it to stop early. That does not damage it or add random error
— it gives you a version that has not finished thinking.

| rounds of thinking | score |
|---|---|
| 1 | 25% |
| 4 | 47% |
| 6 | 68% |
| 8 | 86% |
| 10 | 96% |

One dial, from novice to expert, and it is not a handicap bolted on afterwards.
It is literally how long the opponent considers the problem before answering.
Six rounds is someone eyeballing it; ten is someone doing the sums. Since human
players sit around 70–80%, seven or so makes for a properly close game.

## What I actually learned

**Check what "doing nothing clever" scores, before you believe anything.** The
napkin rule matched the model I was proud of. Ten minutes on the first afternoon
would have saved me years of believing something that was not true — not years of
work, but years of a comfortable illusion nobody had any reason to poke at.

**Match the tool to the shape of the problem, not the shape of the data.** My
data arrived as pictures, so I reached for the picture tool. But the problem was
never really about pictures — it was about everything depending on everything —
and no amount of tuning was going to bridge that.

**A smaller model that is shaped correctly beats a larger one that is not.** The
final version is a seventh the size and thirty-six points better. It was never
short of capacity. It was wired for the wrong job.

And one that I did not expect: the thing that finally worked was not a machine
learning idea at all. It was a numerical method from the 1940s, which I already
knew, and which I had failed to recognise because I was busy thinking about
neural networks instead of about trusses.

## Play it

All three versions run in your browser, no download:
[mkrauter.github.io/TrussGame](https://mkrauter.github.io/TrussGame/) — the
original game, the 2021 model, and the current one. Press **V** while playing
the newest version to see the structure it recovered from the screen, which I
find oddly satisfying to watch.

If you want the actual mathematics — the stiffness matrices, the message
passing, the equilibrium loss, and the experiment where widening the receptive
field made things worse — it is all in
[a notebook](https://colab.research.google.com/github/mkrauter/TrussGame/blob/master/truss_game_v3_training.ipynb)
you can run for free. It builds the training data and trains the model to 96% in
about six minutes, so you can watch it happen rather than take my word for it.

And if you can close the gap between 96.3% and the 98% that eyesight allows, or
get there with fewer parameters — please tell me. Preferably before another five
years go by.
