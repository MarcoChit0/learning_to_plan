; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 918612

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child405 - child
    bread18 - bread-portion
    content309 - content-portion
    tray225 tray461 tray235 tray141 - tray
    table247 table318 table423 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray225 kitchen)
     (at tray461 kitchen)
     (at tray235 kitchen)
     (at tray141 kitchen)
     (at_kitchen_bread bread18)
     (at_kitchen_content content309)
     (not_allergic_gluten child405)
     (waiting child405 table423)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child405)
    )
  )
)
