; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 596945

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child44 child249 - child
    bread402 bread15 - bread-portion
    content355 content321 - content-portion
    tray249 tray454 - tray
    table160 table127 table238 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray249 kitchen)
     (at tray454 kitchen)
     (at_kitchen_bread bread402)
     (at_kitchen_bread bread15)
     (at_kitchen_content content355)
     (at_kitchen_content content321)
     (no_gluten_bread bread402)
     (no_gluten_content content321)
     (allergic_gluten child44)
     (not_allergic_gluten child249)
     (waiting child44 table160)
     (waiting child249 table127)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child44)
     (served child249)
    )
  )
)
