

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a c)
(on b h)
(ontable c)
(on d f)
(on e d)
(on f b)
(ontable g)
(on h a)
(on i g)
(clear e)
(clear i)
)
(:goal
(and
(on a g)
(on b f)
(on c e)
(on d h)
(on f a)
(on g d)
(on h c)
(on i b))
)
)


