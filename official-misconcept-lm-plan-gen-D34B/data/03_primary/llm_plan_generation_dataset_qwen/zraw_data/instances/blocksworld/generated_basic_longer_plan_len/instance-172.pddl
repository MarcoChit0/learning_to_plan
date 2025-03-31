

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(ontable a)
(on b d)
(on c a)
(on d h)
(on e g)
(on f c)
(on g i)
(ontable h)
(on i b)
(clear e)
(clear f)
)
(:goal
(and
(on a b)
(on c f)
(on d h)
(on e i)
(on f a)
(on h g)
(on i c))
)
)


