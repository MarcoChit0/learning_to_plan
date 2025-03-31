; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 738606

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child25 - child
    bread173 - bread-portion
    content451 - content-portion
    tray132 tray128 tray7 - tray
    table400 table370 table186 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray132 kitchen)
     (at tray128 kitchen)
     (at tray7 kitchen)
     (at_kitchen_bread bread173)
     (at_kitchen_content content451)
     (not_allergic_gluten child25)
     (waiting child25 table400)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child25)
    )
  )
)
