; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 182108

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child195 child310 - child
    bread140 bread189 - bread-portion
    content175 content494 - content-portion
    tray84 tray112 tray70 - tray
    table309 table414 table476 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray84 kitchen)
     (at tray112 kitchen)
     (at tray70 kitchen)
     (at_kitchen_bread bread140)
     (at_kitchen_bread bread189)
     (at_kitchen_content content175)
     (at_kitchen_content content494)
     (no_gluten_bread bread140)
     (no_gluten_content content175)
     (allergic_gluten child195)
     (not_allergic_gluten child310)
     (waiting child195 table476)
     (waiting child310 table309)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child195)
     (served child310)
    )
  )
)
