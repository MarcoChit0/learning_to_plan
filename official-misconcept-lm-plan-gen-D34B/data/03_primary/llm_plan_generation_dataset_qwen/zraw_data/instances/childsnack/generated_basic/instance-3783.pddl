; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 370373

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child53 - child
    bread174 - bread-portion
    content125 - content-portion
    tray459 - tray
    table139 table459 table168 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray459 kitchen)
     (at_kitchen_bread bread174)
     (at_kitchen_content content125)
     (not_allergic_gluten child53)
     (waiting child53 table139)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child53)
    )
  )
)
