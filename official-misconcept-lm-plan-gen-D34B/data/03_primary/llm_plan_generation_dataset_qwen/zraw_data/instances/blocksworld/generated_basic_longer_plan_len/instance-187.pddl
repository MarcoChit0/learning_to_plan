

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a f)
(on b h)
(on c d)
(on d i)
(on e a)
(on f g)
(on g b)
(on h c)
(ontable i)
(clear e)
)
(:goal
(and
(on a e)
(on b f)
(on c a)
(on d g)
(on e i)
(on g h)
(on h c)
(on i b))
)
)


