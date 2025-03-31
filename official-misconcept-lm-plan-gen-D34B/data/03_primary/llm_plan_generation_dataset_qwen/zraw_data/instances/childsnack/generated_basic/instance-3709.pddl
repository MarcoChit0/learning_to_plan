; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 698285

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child435 child498 - child
    bread416 bread162 - bread-portion
    content383 content10 - content-portion
    tray127 tray371 - tray
    table481 table173 table200 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray127 kitchen)
     (at tray371 kitchen)
     (at_kitchen_bread bread416)
     (at_kitchen_bread bread162)
     (at_kitchen_content content383)
     (at_kitchen_content content10)
     (no_gluten_bread bread162)
     (no_gluten_content content383)
     (allergic_gluten child435)
     (not_allergic_gluten child498)
     (waiting child435 table481)
     (waiting child498 table481)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child435)
     (served child498)
    )
  )
)
