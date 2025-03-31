; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 563966

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child299 child458 - child
    bread89 bread17 - bread-portion
    content159 content150 - content-portion
    tray348 tray239 tray3 - tray
    table393 table78 table365 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray348 kitchen)
     (at tray239 kitchen)
     (at tray3 kitchen)
     (at_kitchen_bread bread89)
     (at_kitchen_bread bread17)
     (at_kitchen_content content159)
     (at_kitchen_content content150)
     (no_gluten_bread bread89)
     (no_gluten_content content150)
     (allergic_gluten child299)
     (not_allergic_gluten child458)
     (waiting child299 table78)
     (waiting child458 table393)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child299)
     (served child458)
    )
  )
)
