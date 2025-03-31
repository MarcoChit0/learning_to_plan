; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 787079

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child15 child133 - child
    bread331 bread300 - bread-portion
    content471 content159 - content-portion
    tray340 - tray
    table98 table218 table49 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray340 kitchen)
     (at_kitchen_bread bread331)
     (at_kitchen_bread bread300)
     (at_kitchen_content content471)
     (at_kitchen_content content159)
     (no_gluten_bread bread331)
     (no_gluten_content content159)
     (allergic_gluten child15)
     (not_allergic_gluten child133)
     (waiting child15 table98)
     (waiting child133 table218)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child15)
     (served child133)
    )
  )
)
