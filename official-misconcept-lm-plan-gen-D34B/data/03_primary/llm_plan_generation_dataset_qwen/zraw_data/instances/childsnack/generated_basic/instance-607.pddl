; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 136285

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 - child
    bread203 - bread-portion
    content43 - content-portion
    tray24 - tray
    table297 table49 table219 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray24 kitchen)
     (at_kitchen_bread bread203)
     (at_kitchen_content content43)
     (not_allergic_gluten child490)
     (waiting child490 table49)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child490)
    )
  )
)
