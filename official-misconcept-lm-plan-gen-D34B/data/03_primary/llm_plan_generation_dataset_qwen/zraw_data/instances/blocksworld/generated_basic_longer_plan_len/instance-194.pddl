

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i j )
(:init
(handempty)
(on a f)
(ontable b)
(on c j)
(ontable d)
(on e g)
(on f c)
(on g i)
(on h d)
(on i b)
(on j h)
(clear a)
(clear e)
)
(:goal
(and
(on b c)
(on c h)
(on e f)
(on f d)
(on g b)
(on h e)
(on i g)
(on j i))
)
)


