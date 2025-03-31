; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 238840

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child94 child289 - child
    bread219 bread145 - bread-portion
    content288 content112 - content-portion
    tray408 tray478 - tray
    table475 table145 table139 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray408 kitchen)
     (at tray478 kitchen)
     (at_kitchen_bread bread219)
     (at_kitchen_bread bread145)
     (at_kitchen_content content288)
     (at_kitchen_content content112)
     (no_gluten_bread bread219)
     (no_gluten_content content112)
     (allergic_gluten child289)
     (not_allergic_gluten child94)
     (waiting child94 table145)
     (waiting child289 table145)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child94)
     (served child289)
    )
  )
)
