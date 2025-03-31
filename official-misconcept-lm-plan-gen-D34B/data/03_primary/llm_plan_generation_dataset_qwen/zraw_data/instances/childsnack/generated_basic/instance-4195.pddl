; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 488734

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child140 child34 - child
    bread339 bread284 - bread-portion
    content204 content431 - content-portion
    tray309 - tray
    table286 table353 table19 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray309 kitchen)
     (at_kitchen_bread bread339)
     (at_kitchen_bread bread284)
     (at_kitchen_content content204)
     (at_kitchen_content content431)
     (not_allergic_gluten child34)
     (not_allergic_gluten child140)
     (waiting child140 table19)
     (waiting child34 table353)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child140)
     (served child34)
    )
  )
)
