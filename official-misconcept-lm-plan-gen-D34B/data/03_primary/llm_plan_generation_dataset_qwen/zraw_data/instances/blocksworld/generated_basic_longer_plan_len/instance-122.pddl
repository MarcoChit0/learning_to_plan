

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a g)
(on b a)
(on c e)
(ontable d)
(ontable e)
(on f c)
(on g f)
(on h i)
(on i b)
(clear d)
(clear h)
)
(:goal
(and
(on a h)
(on b e)
(on c f)
(on d c)
(on e g)
(on f b)
(on h d))
)
)


