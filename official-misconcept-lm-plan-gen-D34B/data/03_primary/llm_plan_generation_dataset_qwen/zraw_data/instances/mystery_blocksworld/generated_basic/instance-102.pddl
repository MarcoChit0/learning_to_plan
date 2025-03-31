

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b c)
(craves c d)
(craves d a)
(province b)
)
(:goal
(and
(craves a b)
(craves b d))
)
)


