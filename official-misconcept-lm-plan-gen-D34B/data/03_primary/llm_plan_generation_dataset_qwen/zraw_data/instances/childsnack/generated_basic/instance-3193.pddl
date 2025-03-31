; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 338499

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child226 child483 - child
    bread361 bread208 - bread-portion
    content157 content332 - content-portion
    tray242 tray438 - tray
    table452 table30 table310 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray242 kitchen)
     (at tray438 kitchen)
     (at_kitchen_bread bread361)
     (at_kitchen_bread bread208)
     (at_kitchen_content content157)
     (at_kitchen_content content332)
     (no_gluten_bread bread208)
     (no_gluten_content content332)
     (allergic_gluten child226)
     (not_allergic_gluten child483)
     (waiting child226 table452)
     (waiting child483 table30)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child226)
     (served child483)
    )
  )
)
