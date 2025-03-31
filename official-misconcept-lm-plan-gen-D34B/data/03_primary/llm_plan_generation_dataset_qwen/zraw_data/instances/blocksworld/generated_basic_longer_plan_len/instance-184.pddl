

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a h)
(on b g)
(on c b)
(on d i)
(on e f)
(ontable f)
(on g a)
(on h e)
(ontable i)
(clear c)
(clear d)
)
(:goal
(and
(on b g)
(on c a)
(on d e)
(on e f)
(on f b)
(on g i)
(on h d)
(on i c))
)
)


