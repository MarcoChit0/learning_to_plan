(define (problem gripper-2-2-3)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room312 room479 - room
ball133 ball322 ball411 - obj)
(:init
(at-robby robot1 room312)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room312)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball133 room312)
(at ball322 room479)
(at ball411 room312)
)
(:goal
(and
(at ball133 room479)
(at ball322 room479)
(at ball411 room312)
)
)
)
