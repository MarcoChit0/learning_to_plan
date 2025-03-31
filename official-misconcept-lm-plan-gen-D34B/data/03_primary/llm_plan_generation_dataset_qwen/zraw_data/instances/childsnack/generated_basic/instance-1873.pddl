; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 76393

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child96 child249 - child
    bread18 bread96 - bread-portion
    content7 content495 - content-portion
    tray330 tray336 - tray
    table232 table4 table58 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray330 kitchen)
     (at tray336 kitchen)
     (at_kitchen_bread bread18)
     (at_kitchen_bread bread96)
     (at_kitchen_content content7)
     (at_kitchen_content content495)
     (no_gluten_bread bread18)
     (no_gluten_content content7)
     (allergic_gluten child96)
     (not_allergic_gluten child249)
     (waiting child96 table58)
     (waiting child249 table232)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child96)
     (served child249)
    )
  )
)
