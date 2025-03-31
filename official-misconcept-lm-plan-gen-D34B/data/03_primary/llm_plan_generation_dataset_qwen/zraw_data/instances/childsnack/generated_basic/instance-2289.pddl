; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 197135

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child419 child153 - child
    bread49 bread79 - bread-portion
    content190 content115 - content-portion
    tray46 - tray
    table30 table236 table248 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray46 kitchen)
     (at_kitchen_bread bread49)
     (at_kitchen_bread bread79)
     (at_kitchen_content content190)
     (at_kitchen_content content115)
     (no_gluten_bread bread79)
     (no_gluten_content content190)
     (allergic_gluten child153)
     (not_allergic_gluten child419)
     (waiting child419 table248)
     (waiting child153 table248)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child419)
     (served child153)
    )
  )
)
