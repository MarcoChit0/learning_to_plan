

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(ontable a)
(on b e)
(ontable c)
(on d b)
(ontable e)
(on f d)
(on g c)
(on h g)
(on i a)
(clear f)
(clear h)
(clear i)
)
(:goal
(and
(on a e)
(on b i)
(on d a)
(on e c)
(on f d)
(on g h)
(on h b)
(on i f))
)
)


