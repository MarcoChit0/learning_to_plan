; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 81135

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child454 - child
    bread311 - bread-portion
    content394 - content-portion
    tray380 - tray
    table171 table449 table405 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray380 kitchen)
     (at_kitchen_bread bread311)
     (at_kitchen_content content394)
     (not_allergic_gluten child454)
     (waiting child454 table405)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child454)
    )
  )
)
