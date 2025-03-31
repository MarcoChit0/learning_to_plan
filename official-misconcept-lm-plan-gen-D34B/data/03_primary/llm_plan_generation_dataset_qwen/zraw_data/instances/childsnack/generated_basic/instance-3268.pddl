; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 64301

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child126 child203 - child
    bread44 bread429 - bread-portion
    content26 content67 - content-portion
    tray453 tray95 tray460 tray273 - tray
    table223 table26 table251 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray453 kitchen)
     (at tray95 kitchen)
     (at tray460 kitchen)
     (at tray273 kitchen)
     (at_kitchen_bread bread44)
     (at_kitchen_bread bread429)
     (at_kitchen_content content26)
     (at_kitchen_content content67)
     (no_gluten_bread bread44)
     (no_gluten_content content26)
     (allergic_gluten child126)
     (not_allergic_gluten child203)
     (waiting child126 table26)
     (waiting child203 table251)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child126)
     (served child203)
    )
  )
)
