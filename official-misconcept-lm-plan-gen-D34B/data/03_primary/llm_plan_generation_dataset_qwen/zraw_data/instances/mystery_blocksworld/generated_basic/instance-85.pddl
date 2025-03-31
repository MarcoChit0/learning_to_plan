

(define (problem mystery-BW-rand-3)
(:domain mystery-4ops)
(:objects a b c )
(:init
(harmony)
(planet a)
(craves b a)
(planet c)
(province b)
(province c)
)
(:goal
(and
(craves a c)
(craves c b))
)
)


