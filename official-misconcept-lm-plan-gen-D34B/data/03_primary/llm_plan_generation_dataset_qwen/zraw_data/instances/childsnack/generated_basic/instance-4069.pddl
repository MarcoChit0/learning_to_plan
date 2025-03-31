; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 223957

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child177 - child
    bread121 - bread-portion
    content239 - content-portion
    tray6 tray34 tray457 - tray
    table147 table196 table338 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray6 kitchen)
     (at tray34 kitchen)
     (at tray457 kitchen)
     (at_kitchen_bread bread121)
     (at_kitchen_content content239)
     (not_allergic_gluten child177)
     (waiting child177 table196)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child177)
    )
  )
)
