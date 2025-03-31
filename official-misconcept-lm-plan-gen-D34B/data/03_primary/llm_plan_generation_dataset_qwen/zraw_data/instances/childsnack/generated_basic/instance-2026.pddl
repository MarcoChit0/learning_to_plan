; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 93617

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child275 child1 - child
    bread475 bread313 - bread-portion
    content326 content371 - content-portion
    tray178 tray131 tray38 - tray
    table495 table370 table310 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray178 kitchen)
     (at tray131 kitchen)
     (at tray38 kitchen)
     (at_kitchen_bread bread475)
     (at_kitchen_bread bread313)
     (at_kitchen_content content326)
     (at_kitchen_content content371)
     (not_allergic_gluten child1)
     (not_allergic_gluten child275)
     (waiting child275 table370)
     (waiting child1 table370)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child275)
     (served child1)
    )
  )
)
