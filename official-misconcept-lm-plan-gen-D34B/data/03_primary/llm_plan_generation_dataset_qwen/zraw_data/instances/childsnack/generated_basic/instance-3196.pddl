; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 296813

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child134 child222 - child
    bread229 bread209 - bread-portion
    content255 content181 - content-portion
    tray279 tray442 - tray
    table6 table181 table490 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray279 kitchen)
     (at tray442 kitchen)
     (at_kitchen_bread bread229)
     (at_kitchen_bread bread209)
     (at_kitchen_content content255)
     (at_kitchen_content content181)
     (no_gluten_bread bread229)
     (no_gluten_content content181)
     (allergic_gluten child222)
     (not_allergic_gluten child134)
     (waiting child134 table490)
     (waiting child222 table6)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child134)
     (served child222)
    )
  )
)
