; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 991747

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child71 child218 - child
    bread124 bread119 - bread-portion
    content379 content306 - content-portion
    tray176 tray362 tray287 - tray
    table364 table72 table217 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray176 kitchen)
     (at tray362 kitchen)
     (at tray287 kitchen)
     (at_kitchen_bread bread124)
     (at_kitchen_bread bread119)
     (at_kitchen_content content379)
     (at_kitchen_content content306)
     (not_allergic_gluten child218)
     (not_allergic_gluten child71)
     (waiting child71 table364)
     (waiting child218 table217)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child71)
     (served child218)
    )
  )
)
