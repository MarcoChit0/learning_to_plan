; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 731021

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child138 child24 - child
    bread237 bread455 - bread-portion
    content257 content67 - content-portion
    tray205 tray27 tray12 tray350 - tray
    table256 table284 table6 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray205 kitchen)
     (at tray27 kitchen)
     (at tray12 kitchen)
     (at tray350 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_bread bread455)
     (at_kitchen_content content257)
     (at_kitchen_content content67)
     (no_gluten_bread bread237)
     (no_gluten_content content67)
     (allergic_gluten child24)
     (not_allergic_gluten child138)
     (waiting child138 table6)
     (waiting child24 table256)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child138)
     (served child24)
    )
  )
)
