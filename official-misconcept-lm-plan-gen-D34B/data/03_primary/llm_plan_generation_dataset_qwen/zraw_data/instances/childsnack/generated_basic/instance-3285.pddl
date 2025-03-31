; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 797471

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child493 child166 - child
    bread169 bread186 - bread-portion
    content476 content272 - content-portion
    tray105 tray169 tray361 tray368 - tray
    table33 table3 table249 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray105 kitchen)
     (at tray169 kitchen)
     (at tray361 kitchen)
     (at tray368 kitchen)
     (at_kitchen_bread bread169)
     (at_kitchen_bread bread186)
     (at_kitchen_content content476)
     (at_kitchen_content content272)
     (no_gluten_bread bread169)
     (no_gluten_content content272)
     (allergic_gluten child166)
     (not_allergic_gluten child493)
     (waiting child493 table3)
     (waiting child166 table33)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child493)
     (served child166)
    )
  )
)
