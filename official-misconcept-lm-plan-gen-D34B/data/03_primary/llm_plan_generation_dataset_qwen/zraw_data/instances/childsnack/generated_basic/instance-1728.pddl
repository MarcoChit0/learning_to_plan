; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 532130

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child250 - child
    bread432 - bread-portion
    content302 - content-portion
    tray297 tray206 tray366 - tray
    table245 table74 table0 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray297 kitchen)
     (at tray206 kitchen)
     (at tray366 kitchen)
     (at_kitchen_bread bread432)
     (at_kitchen_content content302)
     (not_allergic_gluten child250)
     (waiting child250 table245)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child250)
    )
  )
)
