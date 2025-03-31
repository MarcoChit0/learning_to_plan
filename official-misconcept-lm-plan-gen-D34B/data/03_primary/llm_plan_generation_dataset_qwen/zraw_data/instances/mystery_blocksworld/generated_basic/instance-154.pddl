

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b d)
(craves c a)
(planet d)
(province b)
(province c)
)
(:goal
(and
(craves a b)
(craves b d)
(craves d c))
)
)


