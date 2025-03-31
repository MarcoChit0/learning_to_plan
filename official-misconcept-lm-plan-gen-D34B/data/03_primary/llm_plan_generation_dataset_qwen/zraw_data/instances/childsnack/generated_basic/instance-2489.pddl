; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 134311

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 child41 - child
    bread166 bread428 - bread-portion
    content374 content105 - content-portion
    tray332 tray137 - tray
    table312 table414 table478 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray332 kitchen)
     (at tray137 kitchen)
     (at_kitchen_bread bread166)
     (at_kitchen_bread bread428)
     (at_kitchen_content content374)
     (at_kitchen_content content105)
     (not_allergic_gluten child495)
     (not_allergic_gluten child41)
     (waiting child495 table312)
     (waiting child41 table414)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child495)
     (served child41)
    )
  )
)
