; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 552395

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child165 - child
    bread253 - bread-portion
    content289 - content-portion
    tray297 - tray
    table167 table488 table82 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray297 kitchen)
     (at_kitchen_bread bread253)
     (at_kitchen_content content289)
     (not_allergic_gluten child165)
     (waiting child165 table167)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child165)
    )
  )
)
