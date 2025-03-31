; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 98506

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child132 child367 - child
    bread343 bread351 - bread-portion
    content117 content174 - content-portion
    tray328 - tray
    table429 table432 table258 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray328 kitchen)
     (at_kitchen_bread bread343)
     (at_kitchen_bread bread351)
     (at_kitchen_content content117)
     (at_kitchen_content content174)
     (not_allergic_gluten child132)
     (not_allergic_gluten child367)
     (waiting child132 table432)
     (waiting child367 table432)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child132)
     (served child367)
    )
  )
)
