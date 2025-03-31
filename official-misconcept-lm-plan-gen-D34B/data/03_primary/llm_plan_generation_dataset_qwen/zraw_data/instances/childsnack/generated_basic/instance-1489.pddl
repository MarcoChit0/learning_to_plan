; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 114648

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child103 child340 - child
    bread91 bread49 - bread-portion
    content146 content216 - content-portion
    tray276 tray193 tray401 - tray
    table200 table293 table16 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray276 kitchen)
     (at tray193 kitchen)
     (at tray401 kitchen)
     (at_kitchen_bread bread91)
     (at_kitchen_bread bread49)
     (at_kitchen_content content146)
     (at_kitchen_content content216)
     (no_gluten_bread bread91)
     (no_gluten_content content146)
     (allergic_gluten child103)
     (not_allergic_gluten child340)
     (waiting child103 table16)
     (waiting child340 table16)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child103)
     (served child340)
    )
  )
)
