

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a g)
(on b e)
(on c b)
(ontable d)
(ontable e)
(on f h)
(on g c)
(ontable h)
(clear a)
(clear d)
(clear f)
)
(:goal
(and
(on b f)
(on c e)
(on d h)
(on e a)
(on f g)
(on g d)
(on h c))
)
)


