; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 145305

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child252 child225 - child
    bread227 bread249 - bread-portion
    content173 content27 - content-portion
    tray15 tray286 tray280 - tray
    table190 table118 table347 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray15 kitchen)
     (at tray286 kitchen)
     (at tray280 kitchen)
     (at_kitchen_bread bread227)
     (at_kitchen_bread bread249)
     (at_kitchen_content content173)
     (at_kitchen_content content27)
     (no_gluten_bread bread227)
     (no_gluten_content content27)
     (allergic_gluten child225)
     (not_allergic_gluten child252)
     (waiting child252 table347)
     (waiting child225 table347)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child252)
     (served child225)
    )
  )
)
