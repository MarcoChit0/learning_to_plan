

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a h)
(on b g)
(on c a)
(on d f)
(on e c)
(on f i)
(ontable g)
(on h b)
(on i e)
(clear d)
)
(:goal
(and
(on a h)
(on c i)
(on d c)
(on e f)
(on f g)
(on g b)
(on i a))
)
)


