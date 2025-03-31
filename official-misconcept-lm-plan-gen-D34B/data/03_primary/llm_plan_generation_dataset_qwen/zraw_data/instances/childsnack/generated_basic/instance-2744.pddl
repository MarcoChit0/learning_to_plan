; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 568146

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child112 child165 - child
    bread368 bread83 - bread-portion
    content265 content314 - content-portion
    tray71 - tray
    table9 table116 table22 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray71 kitchen)
     (at_kitchen_bread bread368)
     (at_kitchen_bread bread83)
     (at_kitchen_content content265)
     (at_kitchen_content content314)
     (no_gluten_bread bread83)
     (no_gluten_content content265)
     (allergic_gluten child112)
     (not_allergic_gluten child165)
     (waiting child112 table116)
     (waiting child165 table22)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child112)
     (served child165)
    )
  )
)
