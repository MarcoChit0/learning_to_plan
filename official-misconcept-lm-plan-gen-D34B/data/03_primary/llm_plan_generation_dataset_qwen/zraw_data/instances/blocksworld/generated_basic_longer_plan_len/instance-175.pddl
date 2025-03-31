

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a c)
(on b d)
(on c i)
(on d e)
(on e g)
(ontable f)
(on g h)
(on h f)
(on i b)
(clear a)
)
(:goal
(and
(on a d)
(on b a)
(on c f)
(on d g)
(on e i)
(on h b)
(on i h))
)
)


