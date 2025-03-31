; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 90747

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child292 - child
    bread318 - bread-portion
    content50 - content-portion
    tray220 tray362 tray159 - tray
    table319 table469 table400 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray220 kitchen)
     (at tray362 kitchen)
     (at tray159 kitchen)
     (at_kitchen_bread bread318)
     (at_kitchen_content content50)
     (not_allergic_gluten child292)
     (waiting child292 table319)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child292)
    )
  )
)
