

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a c)
(on b i)
(on c f)
(ontable d)
(on e g)
(on f b)
(on g a)
(on h d)
(on i h)
(clear e)
)
(:goal
(and
(on b e)
(on c h)
(on d c)
(on e i)
(on f a)
(on g b)
(on i d))
)
)


