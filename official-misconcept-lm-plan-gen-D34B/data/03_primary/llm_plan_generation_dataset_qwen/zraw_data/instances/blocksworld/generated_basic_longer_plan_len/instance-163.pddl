

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a d)
(ontable b)
(on c f)
(on d g)
(on e a)
(on f e)
(on g h)
(ontable h)
(on i c)
(clear b)
(clear i)
)
(:goal
(and
(on a f)
(on b c)
(on c d)
(on e g)
(on f b)
(on g a)
(on h i)
(on i e))
)
)


