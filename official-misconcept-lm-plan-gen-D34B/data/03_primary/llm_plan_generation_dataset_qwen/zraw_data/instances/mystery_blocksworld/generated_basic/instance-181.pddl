

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(craves b c)
(craves c a)
(planet d)
(province b)
)
(:goal
(and
(craves b d)
(craves c a)
(craves d c))
)
)


