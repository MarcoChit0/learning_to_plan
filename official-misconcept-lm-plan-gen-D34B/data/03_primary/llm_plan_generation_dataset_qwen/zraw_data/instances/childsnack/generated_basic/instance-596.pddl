; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 856566

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child367 - child
    bread489 - bread-portion
    content345 - content-portion
    tray487 - tray
    table171 table423 table297 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray487 kitchen)
     (at_kitchen_bread bread489)
     (at_kitchen_content content345)
     (not_allergic_gluten child367)
     (waiting child367 table297)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child367)
    )
  )
)
