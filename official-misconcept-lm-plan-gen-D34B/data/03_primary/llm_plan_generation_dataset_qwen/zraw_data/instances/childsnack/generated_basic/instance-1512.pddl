; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 564710

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child100 child469 - child
    bread368 bread284 - bread-portion
    content87 content112 - content-portion
    tray394 tray15 tray471 - tray
    table321 table41 table484 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray394 kitchen)
     (at tray15 kitchen)
     (at tray471 kitchen)
     (at_kitchen_bread bread368)
     (at_kitchen_bread bread284)
     (at_kitchen_content content87)
     (at_kitchen_content content112)
     (no_gluten_bread bread284)
     (no_gluten_content content87)
     (allergic_gluten child100)
     (not_allergic_gluten child469)
     (waiting child100 table321)
     (waiting child469 table321)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child100)
     (served child469)
    )
  )
)
