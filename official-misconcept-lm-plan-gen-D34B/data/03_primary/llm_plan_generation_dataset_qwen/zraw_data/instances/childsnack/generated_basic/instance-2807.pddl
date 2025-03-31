; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 973708

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child56 child268 - child
    bread416 bread73 - bread-portion
    content224 content359 - content-portion
    tray297 tray210 - tray
    table173 table387 table151 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray297 kitchen)
     (at tray210 kitchen)
     (at_kitchen_bread bread416)
     (at_kitchen_bread bread73)
     (at_kitchen_content content224)
     (at_kitchen_content content359)
     (not_allergic_gluten child268)
     (not_allergic_gluten child56)
     (waiting child56 table151)
     (waiting child268 table151)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child56)
     (served child268)
    )
  )
)
