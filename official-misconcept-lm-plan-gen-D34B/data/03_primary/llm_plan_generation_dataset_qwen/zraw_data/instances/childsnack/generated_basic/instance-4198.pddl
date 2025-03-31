; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 827657

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child325 child230 - child
    bread405 bread493 - bread-portion
    content86 content484 - content-portion
    tray9 - tray
    table80 table118 table285 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray9 kitchen)
     (at_kitchen_bread bread405)
     (at_kitchen_bread bread493)
     (at_kitchen_content content86)
     (at_kitchen_content content484)
     (not_allergic_gluten child325)
     (not_allergic_gluten child230)
     (waiting child325 table118)
     (waiting child230 table118)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child325)
     (served child230)
    )
  )
)
