; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 20957

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child233 child262 - child
    bread131 bread143 - bread-portion
    content309 content465 - content-portion
    tray92 tray495 tray456 tray233 - tray
    table177 table107 table460 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray92 kitchen)
     (at tray495 kitchen)
     (at tray456 kitchen)
     (at tray233 kitchen)
     (at_kitchen_bread bread131)
     (at_kitchen_bread bread143)
     (at_kitchen_content content309)
     (at_kitchen_content content465)
     (no_gluten_bread bread143)
     (no_gluten_content content465)
     (allergic_gluten child233)
     (not_allergic_gluten child262)
     (waiting child233 table107)
     (waiting child262 table177)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child233)
     (served child262)
    )
  )
)
