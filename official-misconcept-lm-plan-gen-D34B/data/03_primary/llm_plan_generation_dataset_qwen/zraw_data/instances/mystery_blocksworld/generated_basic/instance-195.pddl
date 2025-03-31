

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(craves b d)
(planet c)
(craves d a)
(province b)
)
(:goal
(and
(craves b c)
(craves c a)
(craves d b))
)
)


