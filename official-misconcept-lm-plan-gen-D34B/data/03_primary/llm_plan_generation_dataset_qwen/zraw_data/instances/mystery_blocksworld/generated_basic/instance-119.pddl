

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b d)
(craves c b)
(craves d a)
(province c)
)
(:goal
(and
(craves a c)
(craves c d)
(craves d b))
)
)


