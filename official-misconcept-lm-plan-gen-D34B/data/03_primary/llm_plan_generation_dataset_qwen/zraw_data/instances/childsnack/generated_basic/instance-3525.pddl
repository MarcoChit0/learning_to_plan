; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 145205

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child405 child70 - child
    bread100 bread221 - bread-portion
    content90 content432 - content-portion
    tray357 tray365 - tray
    table234 table123 table213 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray357 kitchen)
     (at tray365 kitchen)
     (at_kitchen_bread bread100)
     (at_kitchen_bread bread221)
     (at_kitchen_content content90)
     (at_kitchen_content content432)
     (not_allergic_gluten child405)
     (not_allergic_gluten child70)
     (waiting child405 table213)
     (waiting child70 table234)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child405)
     (served child70)
    )
  )
)
