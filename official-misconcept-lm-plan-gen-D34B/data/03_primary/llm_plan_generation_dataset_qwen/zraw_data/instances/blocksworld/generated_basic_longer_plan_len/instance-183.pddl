

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(ontable a)
(on b f)
(on c a)
(on d b)
(on e d)
(on f g)
(on g h)
(on h c)
(on i e)
(clear i)
)
(:goal
(and
(on b f)
(on c b)
(on d i)
(on e c)
(on f a)
(on g d)
(on h e)
(on i h))
)
)


