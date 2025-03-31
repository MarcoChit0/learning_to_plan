; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 173558

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child39 - child
    bread164 - bread-portion
    content351 - content-portion
    tray9 tray382 tray76 - tray
    table172 table336 table208 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray9 kitchen)
     (at tray382 kitchen)
     (at tray76 kitchen)
     (at_kitchen_bread bread164)
     (at_kitchen_content content351)
     (not_allergic_gluten child39)
     (waiting child39 table208)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child39)
    )
  )
)
