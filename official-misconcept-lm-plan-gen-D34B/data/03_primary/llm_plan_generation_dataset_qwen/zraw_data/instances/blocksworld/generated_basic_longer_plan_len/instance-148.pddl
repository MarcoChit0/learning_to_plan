

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a g)
(on b h)
(on c e)
(ontable d)
(on e i)
(on f a)
(on g b)
(ontable h)
(on i f)
(clear c)
(clear d)
)
(:goal
(and
(on a h)
(on c e)
(on e b)
(on f c)
(on g a)
(on h i)
(on i f))
)
)


