; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 676435

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child256 - child
    bread246 - bread-portion
    content196 - content-portion
    tray149 - tray
    table499 table370 table171 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray149 kitchen)
     (at_kitchen_bread bread246)
     (at_kitchen_content content196)
     (not_allergic_gluten child256)
     (waiting child256 table171)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child256)
    )
  )
)
