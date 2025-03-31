; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 597312

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child70 - child
    bread226 - bread-portion
    content108 - content-portion
    tray52 tray301 tray89 tray421 - tray
    table283 table133 table383 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray52 kitchen)
     (at tray301 kitchen)
     (at tray89 kitchen)
     (at tray421 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_content content108)
     (not_allergic_gluten child70)
     (waiting child70 table383)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child70)
    )
  )
)
