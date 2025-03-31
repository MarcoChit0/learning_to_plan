; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 679676

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child202 child0 - child
    bread227 bread151 - bread-portion
    content132 content5 - content-portion
    tray456 tray44 - tray
    table394 table206 table16 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray456 kitchen)
     (at tray44 kitchen)
     (at_kitchen_bread bread227)
     (at_kitchen_bread bread151)
     (at_kitchen_content content132)
     (at_kitchen_content content5)
     (no_gluten_bread bread227)
     (no_gluten_content content132)
     (allergic_gluten child0)
     (not_allergic_gluten child202)
     (waiting child202 table16)
     (waiting child0 table394)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child202)
     (served child0)
    )
  )
)
