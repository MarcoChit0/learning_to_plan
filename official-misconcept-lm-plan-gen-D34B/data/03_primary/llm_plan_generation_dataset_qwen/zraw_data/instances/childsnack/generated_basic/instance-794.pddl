; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 86647

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child117 - child
    bread295 - bread-portion
    content440 - content-portion
    tray208 tray59 - tray
    table80 table198 table447 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray208 kitchen)
     (at tray59 kitchen)
     (at_kitchen_bread bread295)
     (at_kitchen_content content440)
     (not_allergic_gluten child117)
     (waiting child117 table447)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child117)
    )
  )
)
