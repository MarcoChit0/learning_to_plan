

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(ontable a)
(on b a)
(on c d)
(on d i)
(on e h)
(on f c)
(on g f)
(on h g)
(on i b)
(clear e)
)
(:goal
(and
(on a i)
(on b c)
(on c e)
(on e h)
(on f d)
(on h f)
(on i b))
)
)


