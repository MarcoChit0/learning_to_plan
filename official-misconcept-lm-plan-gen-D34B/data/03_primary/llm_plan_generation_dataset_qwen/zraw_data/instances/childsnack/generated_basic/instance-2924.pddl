; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 553803

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child36 child377 - child
    bread282 bread195 - bread-portion
    content227 content206 - content-portion
    tray351 tray88 tray29 - tray
    table344 table182 table354 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray351 kitchen)
     (at tray88 kitchen)
     (at tray29 kitchen)
     (at_kitchen_bread bread282)
     (at_kitchen_bread bread195)
     (at_kitchen_content content227)
     (at_kitchen_content content206)
     (no_gluten_bread bread195)
     (no_gluten_content content227)
     (allergic_gluten child377)
     (not_allergic_gluten child36)
     (waiting child36 table182)
     (waiting child377 table182)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child36)
     (served child377)
    )
  )
)
