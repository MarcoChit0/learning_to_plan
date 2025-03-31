

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a g)
(on b e)
(on c f)
(ontable d)
(on e c)
(on f h)
(on g b)
(on h d)
(clear a)
)
(:goal
(and
(on b g)
(on c a)
(on d c)
(on e b)
(on f e)
(on g h))
)
)


