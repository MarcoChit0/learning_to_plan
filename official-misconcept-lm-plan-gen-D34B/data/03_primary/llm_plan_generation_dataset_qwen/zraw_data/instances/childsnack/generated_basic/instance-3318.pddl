; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 98723

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child295 child41 - child
    bread379 bread284 - bread-portion
    content137 content199 - content-portion
    tray282 tray392 tray423 tray88 - tray
    table478 table303 table493 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray282 kitchen)
     (at tray392 kitchen)
     (at tray423 kitchen)
     (at tray88 kitchen)
     (at_kitchen_bread bread379)
     (at_kitchen_bread bread284)
     (at_kitchen_content content137)
     (at_kitchen_content content199)
     (no_gluten_bread bread379)
     (no_gluten_content content137)
     (allergic_gluten child295)
     (not_allergic_gluten child41)
     (waiting child295 table493)
     (waiting child41 table478)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child295)
     (served child41)
    )
  )
)
