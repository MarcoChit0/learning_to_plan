; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 611532

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child366 child242 child495 - child
    bread69 bread483 bread2 - bread-portion
    content265 content173 content120 - content-portion
    tray237 - tray
    table54 table120 table476 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray237 kitchen)
     (at_kitchen_bread bread69)
     (at_kitchen_bread bread483)
     (at_kitchen_bread bread2)
     (at_kitchen_content content265)
     (at_kitchen_content content173)
     (at_kitchen_content content120)
     (not_allergic_gluten child242)
     (not_allergic_gluten child495)
     (not_allergic_gluten child366)
     (waiting child366 table54)
     (waiting child242 table120)
     (waiting child495 table54)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child366)
     (served child242)
     (served child495)
    )
  )
)
