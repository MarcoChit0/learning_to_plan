; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 245790

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child169 child22 - child
    bread291 bread270 - bread-portion
    content98 content406 - content-portion
    tray118 tray120 tray406 - tray
    table266 table478 table82 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray118 kitchen)
     (at tray120 kitchen)
     (at tray406 kitchen)
     (at_kitchen_bread bread291)
     (at_kitchen_bread bread270)
     (at_kitchen_content content98)
     (at_kitchen_content content406)
     (no_gluten_bread bread291)
     (no_gluten_content content98)
     (allergic_gluten child22)
     (not_allergic_gluten child169)
     (waiting child169 table478)
     (waiting child22 table478)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child169)
     (served child22)
    )
  )
)
