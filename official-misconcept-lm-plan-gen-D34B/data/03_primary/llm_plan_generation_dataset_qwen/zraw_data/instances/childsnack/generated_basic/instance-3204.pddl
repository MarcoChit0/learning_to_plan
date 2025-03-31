; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 12641

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child56 child424 - child
    bread262 bread442 - bread-portion
    content148 content143 - content-portion
    tray374 tray399 - tray
    table378 table312 table234 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray374 kitchen)
     (at tray399 kitchen)
     (at_kitchen_bread bread262)
     (at_kitchen_bread bread442)
     (at_kitchen_content content148)
     (at_kitchen_content content143)
     (no_gluten_bread bread262)
     (no_gluten_content content143)
     (allergic_gluten child56)
     (not_allergic_gluten child424)
     (waiting child56 table234)
     (waiting child424 table312)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child56)
     (served child424)
    )
  )
)
