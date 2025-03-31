; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 619787

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child232 child355 - child
    bread134 bread54 - bread-portion
    content136 content366 - content-portion
    tray215 tray232 - tray
    table338 table96 table109 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray215 kitchen)
     (at tray232 kitchen)
     (at_kitchen_bread bread134)
     (at_kitchen_bread bread54)
     (at_kitchen_content content136)
     (at_kitchen_content content366)
     (not_allergic_gluten child232)
     (not_allergic_gluten child355)
     (waiting child232 table109)
     (waiting child355 table109)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child232)
     (served child355)
    )
  )
)
