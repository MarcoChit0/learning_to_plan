; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 579106

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child205 child16 - child
    bread431 bread246 - bread-portion
    content481 content184 - content-portion
    tray102 tray193 tray232 tray489 - tray
    table356 table479 table7 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray102 kitchen)
     (at tray193 kitchen)
     (at tray232 kitchen)
     (at tray489 kitchen)
     (at_kitchen_bread bread431)
     (at_kitchen_bread bread246)
     (at_kitchen_content content481)
     (at_kitchen_content content184)
     (not_allergic_gluten child205)
     (not_allergic_gluten child16)
     (waiting child205 table479)
     (waiting child16 table7)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child205)
     (served child16)
    )
  )
)
