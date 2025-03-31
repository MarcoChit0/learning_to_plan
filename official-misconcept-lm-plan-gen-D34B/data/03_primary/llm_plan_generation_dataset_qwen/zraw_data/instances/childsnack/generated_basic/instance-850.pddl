; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 746521

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child218 child101 - child
    bread50 bread356 - bread-portion
    content423 content247 - content-portion
    tray333 tray263 - tray
    table469 table492 table406 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray333 kitchen)
     (at tray263 kitchen)
     (at_kitchen_bread bread50)
     (at_kitchen_bread bread356)
     (at_kitchen_content content423)
     (at_kitchen_content content247)
     (no_gluten_bread bread50)
     (no_gluten_content content423)
     (allergic_gluten child218)
     (not_allergic_gluten child101)
     (waiting child218 table406)
     (waiting child101 table492)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child218)
     (served child101)
    )
  )
)
