(define (problem gripper-1-3-1)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room134 room323 room301 - room
ball341 - obj)
(:init
(at-robby robot1 room134)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball341 room323)
)
(:goal
(and
(at ball341 room301)
)
)
)
