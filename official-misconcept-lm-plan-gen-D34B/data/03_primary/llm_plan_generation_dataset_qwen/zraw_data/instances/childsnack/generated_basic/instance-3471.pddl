; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 55198

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child211 child205 - child
    bread28 bread25 - bread-portion
    content465 content299 - content-portion
    tray177 tray428 - tray
    table356 table461 table180 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray177 kitchen)
     (at tray428 kitchen)
     (at_kitchen_bread bread28)
     (at_kitchen_bread bread25)
     (at_kitchen_content content465)
     (at_kitchen_content content299)
     (no_gluten_bread bread28)
     (no_gluten_content content465)
     (allergic_gluten child205)
     (not_allergic_gluten child211)
     (waiting child211 table356)
     (waiting child205 table461)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child211)
     (served child205)
    )
  )
)
