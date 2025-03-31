; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 642823

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child213 child82 - child
    bread492 bread416 - bread-portion
    content145 content290 - content-portion
    tray241 - tray
    table349 table97 table4 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray241 kitchen)
     (at_kitchen_bread bread492)
     (at_kitchen_bread bread416)
     (at_kitchen_content content145)
     (at_kitchen_content content290)
     (no_gluten_bread bread492)
     (no_gluten_content content290)
     (allergic_gluten child82)
     (not_allergic_gluten child213)
     (waiting child213 table349)
     (waiting child82 table97)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child213)
     (served child82)
    )
  )
)
