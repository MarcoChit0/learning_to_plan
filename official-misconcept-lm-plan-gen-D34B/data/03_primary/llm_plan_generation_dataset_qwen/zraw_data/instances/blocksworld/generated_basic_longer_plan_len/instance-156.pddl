

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(ontable a)
(on b d)
(on c g)
(on d c)
(ontable e)
(on f a)
(on g h)
(on h i)
(on i e)
(clear b)
(clear f)
)
(:goal
(and
(on a e)
(on b g)
(on d i)
(on e c)
(on f a)
(on h f)
(on i h))
)
)


