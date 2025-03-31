

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(ontable a)
(on b c)
(on c h)
(on d f)
(on e g)
(on f i)
(on g d)
(on h a)
(on i b)
(clear e)
)
(:goal
(and
(on a d)
(on c b)
(on d e)
(on e f)
(on f i)
(on g c)
(on h g)
(on i h))
)
)


