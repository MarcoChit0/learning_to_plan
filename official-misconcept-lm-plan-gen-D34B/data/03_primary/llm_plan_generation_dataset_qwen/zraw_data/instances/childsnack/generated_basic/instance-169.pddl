; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 165164

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child487 child170 - child
    bread222 bread484 - bread-portion
    content133 content76 - content-portion
    tray403 tray227 tray363 tray149 - tray
    table63 table411 table94 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray403 kitchen)
     (at tray227 kitchen)
     (at tray363 kitchen)
     (at tray149 kitchen)
     (at_kitchen_bread bread222)
     (at_kitchen_bread bread484)
     (at_kitchen_content content133)
     (at_kitchen_content content76)
     (not_allergic_gluten child170)
     (not_allergic_gluten child487)
     (waiting child487 table63)
     (waiting child170 table94)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child487)
     (served child170)
    )
  )
)
