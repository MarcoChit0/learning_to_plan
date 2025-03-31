; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 265375

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child294 - child
    bread85 - bread-portion
    content300 - content-portion
    tray430 tray203 tray484 - tray
    table234 table112 table253 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray430 kitchen)
     (at tray203 kitchen)
     (at tray484 kitchen)
     (at_kitchen_bread bread85)
     (at_kitchen_content content300)
     (not_allergic_gluten child294)
     (waiting child294 table234)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child294)
    )
  )
)
