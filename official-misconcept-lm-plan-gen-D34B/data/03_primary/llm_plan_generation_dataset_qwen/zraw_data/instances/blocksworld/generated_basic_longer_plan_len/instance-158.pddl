

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a g)
(on b d)
(on c h)
(ontable d)
(ontable e)
(on f i)
(ontable g)
(on h f)
(on i e)
(clear a)
(clear b)
(clear c)
)
(:goal
(and
(on a b)
(on b e)
(on c a)
(on d i)
(on e g)
(on f c)
(on g d)
(on h f))
)
)


