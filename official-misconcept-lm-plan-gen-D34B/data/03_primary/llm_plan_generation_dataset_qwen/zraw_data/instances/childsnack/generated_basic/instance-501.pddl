; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 388097

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child405 child459 - child
    bread110 bread192 - bread-portion
    content436 content50 - content-portion
    tray428 tray389 - tray
    table339 table475 table217 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray428 kitchen)
     (at tray389 kitchen)
     (at_kitchen_bread bread110)
     (at_kitchen_bread bread192)
     (at_kitchen_content content436)
     (at_kitchen_content content50)
     (not_allergic_gluten child405)
     (not_allergic_gluten child459)
     (waiting child405 table339)
     (waiting child459 table339)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child405)
     (served child459)
    )
  )
)
