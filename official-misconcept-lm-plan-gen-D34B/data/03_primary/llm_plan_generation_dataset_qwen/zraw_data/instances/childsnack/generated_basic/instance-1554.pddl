; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 976225

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child128 child396 - child
    bread281 bread280 - bread-portion
    content190 content148 - content-portion
    tray109 tray172 tray406 tray214 - tray
    table112 table210 table299 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray109 kitchen)
     (at tray172 kitchen)
     (at tray406 kitchen)
     (at tray214 kitchen)
     (at_kitchen_bread bread281)
     (at_kitchen_bread bread280)
     (at_kitchen_content content190)
     (at_kitchen_content content148)
     (not_allergic_gluten child128)
     (not_allergic_gluten child396)
     (waiting child128 table112)
     (waiting child396 table112)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child128)
     (served child396)
    )
  )
)
