

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(craves b a)
(planet c)
(planet d)
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


