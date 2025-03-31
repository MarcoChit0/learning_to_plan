; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 681930

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child131 child354 - child
    bread389 bread366 - bread-portion
    content216 content148 - content-portion
    tray415 tray26 tray35 - tray
    table113 table327 table293 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray415 kitchen)
     (at tray26 kitchen)
     (at tray35 kitchen)
     (at_kitchen_bread bread389)
     (at_kitchen_bread bread366)
     (at_kitchen_content content216)
     (at_kitchen_content content148)
     (not_allergic_gluten child354)
     (not_allergic_gluten child131)
     (waiting child131 table293)
     (waiting child354 table327)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child131)
     (served child354)
    )
  )
)
