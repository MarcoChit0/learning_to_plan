(define (problem gripper-2-2-5)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room175 room478 - room
ball152 ball268 ball31 ball452 ball378 - obj)
(:init
(at-robby robot1 room478)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room175)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball152 room478)
(at ball268 room478)
(at ball31 room478)
(at ball452 room478)
(at ball378 room175)
)
(:goal
(and
(at ball152 room478)
(at ball268 room175)
(at ball31 room175)
(at ball452 room175)
(at ball378 room478)
)
)
)
