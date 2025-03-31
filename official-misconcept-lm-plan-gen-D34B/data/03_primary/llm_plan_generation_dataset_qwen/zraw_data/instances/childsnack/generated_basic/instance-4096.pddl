; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 349616

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child0 - child
    bread192 - bread-portion
    content72 - content-portion
    tray130 tray79 tray34 - tray
    table258 table482 table436 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray130 kitchen)
     (at tray79 kitchen)
     (at tray34 kitchen)
     (at_kitchen_bread bread192)
     (at_kitchen_content content72)
     (not_allergic_gluten child0)
     (waiting child0 table482)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child0)
    )
  )
)
