; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 422631

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child181 - child
    bread313 - bread-portion
    content285 - content-portion
    tray361 - tray
    table461 table238 table427 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray361 kitchen)
     (at_kitchen_bread bread313)
     (at_kitchen_content content285)
     (not_allergic_gluten child181)
     (waiting child181 table238)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child181)
    )
  )
)
