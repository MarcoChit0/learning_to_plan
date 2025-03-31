; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 608622

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child489 - child
    bread424 - bread-portion
    content232 - content-portion
    tray381 - tray
    table118 table74 table371 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray381 kitchen)
     (at_kitchen_bread bread424)
     (at_kitchen_content content232)
     (not_allergic_gluten child489)
     (waiting child489 table371)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child489)
    )
  )
)
