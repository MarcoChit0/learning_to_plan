

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a d)
(ontable b)
(on c f)
(on d i)
(on e g)
(on f a)
(on g c)
(on h e)
(on i b)
(clear h)
)
(:goal
(and
(on c b)
(on d e)
(on e c)
(on f a)
(on g d)
(on h g)
(on i f))
)
)


