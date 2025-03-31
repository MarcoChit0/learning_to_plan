; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 328017

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 - child
    bread214 - bread-portion
    content12 - content-portion
    tray128 tray179 tray395 tray258 - tray
    table475 table272 table499 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray128 kitchen)
     (at tray179 kitchen)
     (at tray395 kitchen)
     (at tray258 kitchen)
     (at_kitchen_bread bread214)
     (at_kitchen_content content12)
     (not_allergic_gluten child109)
     (waiting child109 table272)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child109)
    )
  )
)
