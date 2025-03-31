

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a g)
(on b h)
(on c f)
(on d a)
(on e i)
(on f b)
(on g c)
(ontable h)
(on i d)
(clear e)
)
(:goal
(and
(on a d)
(on b e)
(on d h)
(on e g)
(on f c)
(on g a)
(on h f)
(on i b))
)
)


