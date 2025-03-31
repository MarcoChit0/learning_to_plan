; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 142588

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child5 child138 - child
    bread473 bread412 - bread-portion
    content130 content474 - content-portion
    tray169 tray74 tray35 tray481 - tray
    table77 table14 table35 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray169 kitchen)
     (at tray74 kitchen)
     (at tray35 kitchen)
     (at tray481 kitchen)
     (at_kitchen_bread bread473)
     (at_kitchen_bread bread412)
     (at_kitchen_content content130)
     (at_kitchen_content content474)
     (no_gluten_bread bread473)
     (no_gluten_content content130)
     (allergic_gluten child138)
     (not_allergic_gluten child5)
     (waiting child5 table35)
     (waiting child138 table35)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child5)
     (served child138)
    )
  )
)
