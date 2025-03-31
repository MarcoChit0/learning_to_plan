; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 685419

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child418 child127 - child
    bread77 bread376 - bread-portion
    content3 content273 - content-portion
    tray471 tray372 - tray
    table172 table163 table243 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray471 kitchen)
     (at tray372 kitchen)
     (at_kitchen_bread bread77)
     (at_kitchen_bread bread376)
     (at_kitchen_content content3)
     (at_kitchen_content content273)
     (not_allergic_gluten child418)
     (not_allergic_gluten child127)
     (waiting child418 table243)
     (waiting child127 table243)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child418)
     (served child127)
    )
  )
)
