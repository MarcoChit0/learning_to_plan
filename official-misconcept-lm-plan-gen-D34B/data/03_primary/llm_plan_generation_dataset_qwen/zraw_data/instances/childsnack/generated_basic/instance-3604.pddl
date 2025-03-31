; child-snack task with 3 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 918614

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child36 child285 child369 - child
    bread95 bread198 bread202 - bread-portion
    content15 content463 content366 - content-portion
    tray237 tray394 tray310 - tray
    table55 table120 table159 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray237 kitchen)
     (at tray394 kitchen)
     (at tray310 kitchen)
     (at_kitchen_bread bread95)
     (at_kitchen_bread bread198)
     (at_kitchen_bread bread202)
     (at_kitchen_content content15)
     (at_kitchen_content content463)
     (at_kitchen_content content366)
     (no_gluten_bread bread95)
     (no_gluten_content content15)
     (allergic_gluten child36)
     (not_allergic_gluten child285)
     (not_allergic_gluten child369)
     (waiting child36 table120)
     (waiting child285 table55)
     (waiting child369 table55)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child36)
     (served child285)
     (served child369)
    )
  )
)
