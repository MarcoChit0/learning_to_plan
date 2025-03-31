

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a c)
(on b d)
(on c h)
(ontable d)
(on e f)
(on f g)
(on g i)
(on h b)
(on i a)
(clear e)
)
(:goal
(and
(on a b)
(on b f)
(on c a)
(on e h)
(on f g)
(on g e)
(on h d))
)
)


