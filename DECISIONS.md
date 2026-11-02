->DECISIONS

Time spent: [3 days - 3-4 hours daily]

-> What I chose to run, and why

I built one small Python program that does everything: it makes the fake belt video, finds and tracks the parcels, counts them, and shows the result on a web page. I used only two libraries (numpy and OpenCV) and Python's built-in web server. There is no database, no queue and no other service, so one command starts the whole thing.

To count, I find the parcels by comparing each frame with a picture of the empty belt. Then I follow each parcel from frame to frame. All parcels on a belt move at the same speed, so I measure that one belt speed and move every parcel by it. This means a parcel that is hidden behind another one can still be followed, and it keeps its ID. A parcel is counted once, when its centre crosses a line on the belt. Because the count depends on the parcel and the line, it does not depend on frame rate or how long a parcel stays in view.

I chose this over a heavy tracker because the parcels are plain coloured boxes seen from above, and this runs about 7 times faster than real time on a small 2-core machine.

-> What it cost

Setup was cheap: two libraries and one process. The hard part was not the wiring, it was parcels that arrive touching each other. My first version counted them as one parcel and missed 34 parcels. Fixing it took a few rounds and added a step that cuts a joined blob into separate parcels by colour. What remains is a set of about a dozen tuned numbers (thresholds) in one file, and a few assumptions listed below.

 -> What I considered and rejected

- Counting blobs in each frame, or counting every new blob: the count would depend on frame rate and how long a parcel stays visible, and two overlapping parcels would count as one.
- Ready-made trackers (SORT / ByteTrack style): when two parcels overlap they look like one box, so one track is lost and comes back later with a new ID. That is a double count.
- Deep-learning detectors or appearance-matching trackers (YOLO, DeepSORT): too heavy for plain coloured boxes on a laptop, and there is no training data for synthetic boxes.
- Measuring belt speed from the belt's texture: a good idea for real footage, but my fake belt has no moving texture, so it would show zero.
- Kalman filters with Hungarian matching: all parcels share one speed and there are only about 10 on screen, so simple matching is enough.

-> The tradeoff I settled on (how many frames to look at)

I look at every frame. Checking more frames keeps each parcel's identity safer but costs more work. Skipping frames is cheaper but can lose track of which parcel is which.

I tested on three new videos that I had not tuned on, counting every 1st, 2nd, 3rd, 4th, 6th and 8th frame:
- Every frame: counted 3442 against a true 3459 (-0.49%). Steady stretches -0.19%, surges -0.62%. Nothing was counted while the belt was stopped. 98.2% of parcels were counted exactly once.
- Every 2nd to 4th frame: almost the same (net error between -0.4% and -0.8%).
- Every 6th frame: it starts to fail (steady error -2.9%, 13 wrong counts while the belt was stopped, three times as many ID switches).
- Every 8th frame: it breaks (-12%).

Why: in a surge a parcel moves 12 pixels per frame, so looking at every 6th frame moves it 72 pixels between looks. That is more than the gap between parcels (about 60 pixels), so parcels get mixed up. Also, an overlap lasts only 3 to 5 frames, so skipping 4 or more frames can jump over it completely.

Cost in each direction: looking at every frame is the most work (about 4 ms per frame, so there is plenty of room at 30 frames per second). Skipping frames saves work but is less safe in surges. Under heavy load I would go to every 2nd frame, and never past every 4th.

-> Assumptions I made

- The belt background and the camera do not move. A real belt with moving marks would break this first.
- All parcels move at the belt's speed, except for short 3 to 5 frame slips that I added to create overlaps.
- Parcels are boxes, enter from the left, and are no bigger than 96 x 72 pixels.
- Overlaps in my video last 3 to 5 frames and hide part of a parcel, not most of it. I did not test a parcel hidden for most of its trip, so I do not claim that case. The closest test: parcels that were at least 70% hidden at some point were still counted once in 56 of 61 cases.
- "True count" means the parcels my generator logged as crossing the line.

-> What I would do next

With more time I would first make the belt background move like a real belt and measure again, because that is the assumption most likely to fail on real footage. Next I would improve how touching parcels are separated, so it also works when their colours are similar, and allow a lost parcel to be picked up again in the middle of the belt (now new parcels can only start near the entry edge). Then I would add the "hidden for most of the trip" parcel to the video and report how often it is counted correctly versus lost.

- Moving belt background, then re-measure.
- Better separation of touching parcels of similar colour.
- Allow recovering a lost parcel mid-belt.
- Add the mostly-hidden parcel case and report it as a number.
- Replace the per-frame text logs with a compact binary file for hour-long runs.

If every number in the brief went 10 times higher, the first thing to break is speed: there would be about 100 parcels on screen and the per-frame work would grow past real time. Then the picture itself would break, because at that density nearly everything is one joined blob. I would switch to a learned detector, split the work across lanes or processes, lower the resolution, and skip frames when busy.
