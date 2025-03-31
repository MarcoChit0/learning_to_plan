; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 546669

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child6 - child
    bread132 - bread-portion
    content108 - content-portion
    tray193 tray173 tray262 tray196 - tray
    table484 table204 table327 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray193 kitchen)
     (at tray173 kitchen)
     (at tray262 kitchen)
     (at tray196 kitchen)
     (at_kitchen_bread bread132)
     (at_kitchen_content content108)
     (not_allergic_gluten child6)
     (waiting child6 table327)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child6)
    )
  )
)
