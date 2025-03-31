; child-snack task with 2 children and 0.56 gluten factor 
; constant factor of 1.3
; random seed: 934201

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child366 child481 - child
    bread192 bread122 - bread-portion
    content12 content73 - content-portion
    tray312 tray131 - tray
    table422 table274 table375 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray312 kitchen)
     (at tray131 kitchen)
     (at_kitchen_bread bread192)
     (at_kitchen_bread bread122)
     (at_kitchen_content content12)
     (at_kitchen_content content73)
     (no_gluten_bread bread122)
     (no_gluten_content content12)
     (allergic_gluten child366)
     (not_allergic_gluten child481)
     (waiting child366 table422)
     (waiting child481 table375)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child366)
     (served child481)
    )
  )
)
