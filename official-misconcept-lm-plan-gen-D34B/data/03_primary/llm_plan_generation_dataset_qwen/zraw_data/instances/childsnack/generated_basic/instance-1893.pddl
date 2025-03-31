; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 262342

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child80 child209 - child
    bread16 bread61 - bread-portion
    content407 content52 - content-portion
    tray486 tray222 - tray
    table248 table271 table478 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray486 kitchen)
     (at tray222 kitchen)
     (at_kitchen_bread bread16)
     (at_kitchen_bread bread61)
     (at_kitchen_content content407)
     (at_kitchen_content content52)
     (no_gluten_bread bread16)
     (no_gluten_content content407)
     (allergic_gluten child80)
     (not_allergic_gluten child209)
     (waiting child80 table248)
     (waiting child209 table248)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child80)
     (served child209)
    )
  )
)
