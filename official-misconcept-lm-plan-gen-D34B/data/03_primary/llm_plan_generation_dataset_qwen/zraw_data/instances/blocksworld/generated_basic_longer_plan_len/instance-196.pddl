

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i j )
(:init
(handempty)
(on a j)
(on b i)
(on c g)
(ontable d)
(on e h)
(on f e)
(ontable g)
(on h b)
(on i d)
(on j c)
(clear a)
(clear f)
)
(:goal
(and
(on a j)
(on c a)
(on d h)
(on e i)
(on f b)
(on h f)
(on j g))
)
)


