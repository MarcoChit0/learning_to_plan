; child-snack task with 2 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 311706

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child427 child149 - child
    bread333 bread129 - bread-portion
    content484 content211 - content-portion
    tray382 - tray
    table363 table88 table489 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray382 kitchen)
     (at_kitchen_bread bread333)
     (at_kitchen_bread bread129)
     (at_kitchen_content content484)
     (at_kitchen_content content211)
     (no_gluten_bread bread129)
     (no_gluten_content content211)
     (allergic_gluten child427)
     (not_allergic_gluten child149)
     (waiting child427 table88)
     (waiting child149 table363)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child427)
     (served child149)
    )
  )
)
