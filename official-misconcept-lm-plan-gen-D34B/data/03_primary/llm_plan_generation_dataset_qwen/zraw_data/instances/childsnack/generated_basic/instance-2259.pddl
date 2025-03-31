; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 176187

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child266 child457 - child
    bread291 bread18 - bread-portion
    content253 content385 - content-portion
    tray233 - tray
    table259 table177 table481 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray233 kitchen)
     (at_kitchen_bread bread291)
     (at_kitchen_bread bread18)
     (at_kitchen_content content253)
     (at_kitchen_content content385)
     (no_gluten_bread bread18)
     (no_gluten_content content253)
     (allergic_gluten child266)
     (not_allergic_gluten child457)
     (waiting child266 table481)
     (waiting child457 table259)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child266)
     (served child457)
    )
  )
)
