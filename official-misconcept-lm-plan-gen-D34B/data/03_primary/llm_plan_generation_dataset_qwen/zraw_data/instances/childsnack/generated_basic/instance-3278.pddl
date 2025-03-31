; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 863835

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child432 child96 - child
    bread316 bread431 - bread-portion
    content83 content16 - content-portion
    tray149 tray290 tray350 tray334 - tray
    table344 table280 table33 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray149 kitchen)
     (at tray290 kitchen)
     (at tray350 kitchen)
     (at tray334 kitchen)
     (at_kitchen_bread bread316)
     (at_kitchen_bread bread431)
     (at_kitchen_content content83)
     (at_kitchen_content content16)
     (no_gluten_bread bread431)
     (no_gluten_content content83)
     (allergic_gluten child96)
     (not_allergic_gluten child432)
     (waiting child432 table344)
     (waiting child96 table33)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child432)
     (served child96)
    )
  )
)
