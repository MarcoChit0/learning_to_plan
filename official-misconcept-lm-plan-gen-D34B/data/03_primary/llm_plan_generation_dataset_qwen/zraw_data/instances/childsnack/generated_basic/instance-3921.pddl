; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 470391

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child484 child56 - child
    bread5 bread243 - bread-portion
    content52 content494 - content-portion
    tray366 - tray
    table172 table160 table452 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray366 kitchen)
     (at_kitchen_bread bread5)
     (at_kitchen_bread bread243)
     (at_kitchen_content content52)
     (at_kitchen_content content494)
     (no_gluten_bread bread243)
     (no_gluten_content content494)
     (allergic_gluten child484)
     (not_allergic_gluten child56)
     (waiting child484 table172)
     (waiting child56 table160)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child484)
     (served child56)
    )
  )
)
