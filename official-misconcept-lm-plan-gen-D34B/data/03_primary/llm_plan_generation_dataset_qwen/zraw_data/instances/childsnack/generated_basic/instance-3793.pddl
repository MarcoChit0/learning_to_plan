; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 555068

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child17 - child
    bread123 - bread-portion
    content301 - content-portion
    tray26 - tray
    table302 table211 table19 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray26 kitchen)
     (at_kitchen_bread bread123)
     (at_kitchen_content content301)
     (not_allergic_gluten child17)
     (waiting child17 table211)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child17)
    )
  )
)
