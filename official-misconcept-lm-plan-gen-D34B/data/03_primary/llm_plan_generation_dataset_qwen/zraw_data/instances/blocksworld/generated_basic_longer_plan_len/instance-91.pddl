

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a h)
(ontable b)
(on c d)
(ontable d)
(on e a)
(on f e)
(on g f)
(on h b)
(clear c)
(clear g)
)
(:goal
(and
(on a d)
(on b f)
(on c h)
(on d b)
(on e c)
(on h a))
)
)


