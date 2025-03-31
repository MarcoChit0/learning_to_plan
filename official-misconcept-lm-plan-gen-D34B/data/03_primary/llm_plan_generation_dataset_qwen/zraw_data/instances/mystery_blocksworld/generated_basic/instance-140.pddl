

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b c)
(planet c)
(craves d a)
(province b)
(province d)
)
(:goal
(and
(craves a c)
(craves b a)
(craves c d))
)
)


