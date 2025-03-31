; child-snack task with 2 children and 0.56 gluten factor 
; constant factor of 1.3
; random seed: 592819

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child45 child132 - child
    bread324 bread214 - bread-portion
    content304 content224 - content-portion
    tray464 tray333 - tray
    table205 table337 table140 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray464 kitchen)
     (at tray333 kitchen)
     (at_kitchen_bread bread324)
     (at_kitchen_bread bread214)
     (at_kitchen_content content304)
     (at_kitchen_content content224)
     (no_gluten_bread bread324)
     (no_gluten_content content224)
     (allergic_gluten child132)
     (not_allergic_gluten child45)
     (waiting child45 table205)
     (waiting child132 table337)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child45)
     (served child132)
    )
  )
)
