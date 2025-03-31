; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 476876

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child120 - child
    bread307 - bread-portion
    content153 - content-portion
    tray245 tray303 tray473 tray33 - tray
    table445 table84 table229 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray245 kitchen)
     (at tray303 kitchen)
     (at tray473 kitchen)
     (at tray33 kitchen)
     (at_kitchen_bread bread307)
     (at_kitchen_content content153)
     (not_allergic_gluten child120)
     (waiting child120 table84)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child120)
    )
  )
)
