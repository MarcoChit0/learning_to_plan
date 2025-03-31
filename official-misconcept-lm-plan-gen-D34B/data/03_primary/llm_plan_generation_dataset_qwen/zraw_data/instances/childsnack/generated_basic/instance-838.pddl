; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 515836

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child100 child165 - child
    bread0 bread479 - bread-portion
    content319 content384 - content-portion
    tray196 tray38 - tray
    table488 table279 table86 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray196 kitchen)
     (at tray38 kitchen)
     (at_kitchen_bread bread0)
     (at_kitchen_bread bread479)
     (at_kitchen_content content319)
     (at_kitchen_content content384)
     (no_gluten_bread bread479)
     (no_gluten_content content384)
     (allergic_gluten child165)
     (not_allergic_gluten child100)
     (waiting child100 table279)
     (waiting child165 table86)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child100)
     (served child165)
    )
  )
)
