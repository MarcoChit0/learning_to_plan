; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 806184

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child456 child242 - child
    bread77 bread402 - bread-portion
    content258 content26 - content-portion
    tray243 tray271 tray337 - tray
    table102 table284 table259 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray243 kitchen)
     (at tray271 kitchen)
     (at tray337 kitchen)
     (at_kitchen_bread bread77)
     (at_kitchen_bread bread402)
     (at_kitchen_content content258)
     (at_kitchen_content content26)
     (no_gluten_bread bread402)
     (no_gluten_content content26)
     (allergic_gluten child456)
     (not_allergic_gluten child242)
     (waiting child456 table259)
     (waiting child242 table259)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child456)
     (served child242)
    )
  )
)
