; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 583665

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child233 child208 - child
    bread101 bread430 - bread-portion
    content80 content316 - content-portion
    tray466 tray169 tray452 tray471 - tray
    table94 table288 table222 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray466 kitchen)
     (at tray169 kitchen)
     (at tray452 kitchen)
     (at tray471 kitchen)
     (at_kitchen_bread bread101)
     (at_kitchen_bread bread430)
     (at_kitchen_content content80)
     (at_kitchen_content content316)
     (not_allergic_gluten child233)
     (not_allergic_gluten child208)
     (waiting child233 table288)
     (waiting child208 table288)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child233)
     (served child208)
    )
  )
)
