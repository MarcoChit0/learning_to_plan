; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 604436

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 - child
    bread297 - bread-portion
    content50 - content-portion
    tray414 - tray
    table140 table66 table290 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray414 kitchen)
     (at_kitchen_bread bread297)
     (at_kitchen_content content50)
     (not_allergic_gluten child431)
     (waiting child431 table66)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child431)
    )
  )
)
