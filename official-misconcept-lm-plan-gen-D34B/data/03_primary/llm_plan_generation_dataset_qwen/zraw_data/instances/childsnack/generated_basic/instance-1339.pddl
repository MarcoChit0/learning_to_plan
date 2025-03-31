; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 84269

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child260 child124 - child
    bread401 bread47 - bread-portion
    content446 content309 - content-portion
    tray419 tray227 tray407 - tray
    table190 table302 table373 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray419 kitchen)
     (at tray227 kitchen)
     (at tray407 kitchen)
     (at_kitchen_bread bread401)
     (at_kitchen_bread bread47)
     (at_kitchen_content content446)
     (at_kitchen_content content309)
     (not_allergic_gluten child124)
     (not_allergic_gluten child260)
     (waiting child260 table373)
     (waiting child124 table190)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child260)
     (served child124)
    )
  )
)
