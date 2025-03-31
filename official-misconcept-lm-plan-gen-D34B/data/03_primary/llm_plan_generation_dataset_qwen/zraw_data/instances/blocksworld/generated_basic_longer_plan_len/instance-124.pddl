

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a h)
(on b e)
(ontable c)
(on d a)
(on e f)
(on f c)
(on g d)
(on h b)
(ontable i)
(clear g)
(clear i)
)
(:goal
(and
(on a f)
(on c a)
(on d c)
(on e i)
(on f b)
(on g h)
(on h d)
(on i g))
)
)


