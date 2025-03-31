(define (problem gripper-2-2-3)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room105 room411 - room
ball445 ball170 ball305 - obj)
(:init
(at-robby robot1 room105)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room411)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball445 room411)
(at ball170 room411)
(at ball305 room411)
)
(:goal
(and
(at ball445 room105)
(at ball170 room411)
(at ball305 room411)
)
)
)
