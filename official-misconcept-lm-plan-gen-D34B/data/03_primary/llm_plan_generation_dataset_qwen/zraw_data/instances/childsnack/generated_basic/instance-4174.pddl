; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 275891

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child330 child235 - child
    bread374 bread367 - bread-portion
    content315 content66 - content-portion
    tray394 - tray
    table22 table262 table397 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray394 kitchen)
     (at_kitchen_bread bread374)
     (at_kitchen_bread bread367)
     (at_kitchen_content content315)
     (at_kitchen_content content66)
     (not_allergic_gluten child330)
     (not_allergic_gluten child235)
     (waiting child330 table262)
     (waiting child235 table262)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child330)
     (served child235)
    )
  )
)
