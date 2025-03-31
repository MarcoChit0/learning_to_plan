; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 151767

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child477 child416 - child
    bread417 bread205 - bread-portion
    content101 content266 - content-portion
    tray3 tray378 tray89 - tray
    table465 table497 table75 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray3 kitchen)
     (at tray378 kitchen)
     (at tray89 kitchen)
     (at_kitchen_bread bread417)
     (at_kitchen_bread bread205)
     (at_kitchen_content content101)
     (at_kitchen_content content266)
     (not_allergic_gluten child416)
     (not_allergic_gluten child477)
     (waiting child477 table465)
     (waiting child416 table465)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child477)
     (served child416)
    )
  )
)
