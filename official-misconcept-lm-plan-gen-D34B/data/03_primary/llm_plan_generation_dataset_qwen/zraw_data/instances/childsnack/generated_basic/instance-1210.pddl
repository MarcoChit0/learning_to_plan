; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 375484

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child436 child472 child480 - child
    bread482 bread124 bread404 - bread-portion
    content222 content399 content150 - content-portion
    tray452 tray285 tray107 - tray
    table50 table27 table425 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray452 kitchen)
     (at tray285 kitchen)
     (at tray107 kitchen)
     (at_kitchen_bread bread482)
     (at_kitchen_bread bread124)
     (at_kitchen_bread bread404)
     (at_kitchen_content content222)
     (at_kitchen_content content399)
     (at_kitchen_content content150)
     (not_allergic_gluten child436)
     (not_allergic_gluten child472)
     (not_allergic_gluten child480)
     (waiting child436 table50)
     (waiting child472 table50)
     (waiting child480 table425)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child436)
     (served child472)
     (served child480)
    )
  )
)
