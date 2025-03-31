; child-snack task with 2 children and 0.51 gluten factor 
; constant factor of 1.9
; random seed: 699880

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child232 child8 - child
    bread218 bread224 - bread-portion
    content142 content401 - content-portion
    tray17 tray244 tray103 - tray
    table310 table401 table298 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray17 kitchen)
     (at tray244 kitchen)
     (at tray103 kitchen)
     (at_kitchen_bread bread218)
     (at_kitchen_bread bread224)
     (at_kitchen_content content142)
     (at_kitchen_content content401)
     (no_gluten_bread bread224)
     (no_gluten_content content401)
     (allergic_gluten child8)
     (not_allergic_gluten child232)
     (waiting child232 table310)
     (waiting child8 table298)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child232)
     (served child8)
    )
  )
)
