

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a g)
(on b c)
(on c e)
(on d i)
(on e a)
(on f b)
(on g h)
(on h d)
(ontable i)
(clear f)
)
(:goal
(and
(on a i)
(on b f)
(on c b)
(on d h)
(on e g)
(on f e)
(on h c)
(on i d))
)
)


