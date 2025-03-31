; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 994833

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child327 - child
    bread482 - bread-portion
    content301 - content-portion
    tray476 tray497 tray480 - tray
    table351 table219 table160 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray476 kitchen)
     (at tray497 kitchen)
     (at tray480 kitchen)
     (at_kitchen_bread bread482)
     (at_kitchen_content content301)
     (not_allergic_gluten child327)
     (waiting child327 table160)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child327)
    )
  )
)
