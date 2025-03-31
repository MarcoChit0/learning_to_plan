; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 903052

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child131 - child
    bread479 - bread-portion
    content25 - content-portion
    tray448 tray244 tray285 tray455 - tray
    table410 table58 table47 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray448 kitchen)
     (at tray244 kitchen)
     (at tray285 kitchen)
     (at tray455 kitchen)
     (at_kitchen_bread bread479)
     (at_kitchen_content content25)
     (not_allergic_gluten child131)
     (waiting child131 table58)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child131)
    )
  )
)
