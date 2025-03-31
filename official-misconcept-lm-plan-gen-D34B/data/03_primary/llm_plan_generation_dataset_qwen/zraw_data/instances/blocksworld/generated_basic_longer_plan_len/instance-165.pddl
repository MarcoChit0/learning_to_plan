

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a d)
(ontable b)
(on c g)
(on d e)
(on e f)
(on f h)
(on g a)
(ontable h)
(on i c)
(clear b)
(clear i)
)
(:goal
(and
(on a d)
(on b a)
(on c h)
(on d c)
(on e f)
(on f b)
(on h i))
)
)


