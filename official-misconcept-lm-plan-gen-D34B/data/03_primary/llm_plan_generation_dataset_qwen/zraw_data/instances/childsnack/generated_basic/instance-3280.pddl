; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 303537

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child72 child3 - child
    bread79 bread496 - bread-portion
    content148 content101 - content-portion
    tray37 tray395 tray241 tray283 - tray
    table356 table494 table458 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray37 kitchen)
     (at tray395 kitchen)
     (at tray241 kitchen)
     (at tray283 kitchen)
     (at_kitchen_bread bread79)
     (at_kitchen_bread bread496)
     (at_kitchen_content content148)
     (at_kitchen_content content101)
     (no_gluten_bread bread79)
     (no_gluten_content content101)
     (allergic_gluten child3)
     (not_allergic_gluten child72)
     (waiting child72 table494)
     (waiting child3 table356)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child72)
     (served child3)
    )
  )
)
