; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 749703

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child126 child70 - child
    bread8 bread421 - bread-portion
    content498 content132 - content-portion
    tray76 - tray
    table144 table43 table354 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray76 kitchen)
     (at_kitchen_bread bread8)
     (at_kitchen_bread bread421)
     (at_kitchen_content content498)
     (at_kitchen_content content132)
     (no_gluten_bread bread8)
     (no_gluten_content content132)
     (allergic_gluten child70)
     (not_allergic_gluten child126)
     (waiting child126 table43)
     (waiting child70 table354)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child126)
     (served child70)
    )
  )
)
