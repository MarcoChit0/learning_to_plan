

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(ontable a)
(on b c)
(on c i)
(on d e)
(on e h)
(on f a)
(on g d)
(on h f)
(ontable i)
(clear b)
(clear g)
)
(:goal
(and
(on a d)
(on b f)
(on d c)
(on e i)
(on f a)
(on g b)
(on h g)
(on i h))
)
)


