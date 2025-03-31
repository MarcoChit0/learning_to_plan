

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a h)
(ontable b)
(on c i)
(ontable d)
(on e g)
(on f c)
(ontable g)
(on h f)
(on i e)
(clear a)
(clear b)
(clear d)
)
(:goal
(and
(on a e)
(on b a)
(on c i)
(on d g)
(on e h)
(on f c)
(on h f)
(on i d))
)
)


