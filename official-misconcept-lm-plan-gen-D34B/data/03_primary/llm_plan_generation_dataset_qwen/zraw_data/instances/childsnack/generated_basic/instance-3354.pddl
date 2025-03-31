; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 189383

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child188 child399 - child
    bread195 bread68 - bread-portion
    content369 content329 - content-portion
    tray486 tray245 - tray
    table319 table478 table15 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray486 kitchen)
     (at tray245 kitchen)
     (at_kitchen_bread bread195)
     (at_kitchen_bread bread68)
     (at_kitchen_content content369)
     (at_kitchen_content content329)
     (no_gluten_bread bread68)
     (no_gluten_content content329)
     (allergic_gluten child399)
     (not_allergic_gluten child188)
     (waiting child188 table478)
     (waiting child399 table319)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child188)
     (served child399)
    )
  )
)
