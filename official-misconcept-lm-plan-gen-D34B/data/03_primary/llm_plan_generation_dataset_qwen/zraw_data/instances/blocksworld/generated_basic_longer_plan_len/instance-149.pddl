

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a i)
(on b a)
(on c g)
(on d h)
(ontable e)
(on f c)
(ontable g)
(on h e)
(on i d)
(clear b)
(clear f)
)
(:goal
(and
(on a f)
(on d i)
(on e g)
(on f e)
(on g b)
(on h a)
(on i h))
)
)


