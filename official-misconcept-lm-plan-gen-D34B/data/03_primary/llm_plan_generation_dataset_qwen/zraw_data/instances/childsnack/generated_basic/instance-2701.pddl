; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 529602

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child14 child160 child449 - child
    bread424 bread198 bread405 - bread-portion
    content494 content113 content230 - content-portion
    tray336 tray373 tray224 - tray
    table232 table275 table399 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray336 kitchen)
     (at tray373 kitchen)
     (at tray224 kitchen)
     (at_kitchen_bread bread424)
     (at_kitchen_bread bread198)
     (at_kitchen_bread bread405)
     (at_kitchen_content content494)
     (at_kitchen_content content113)
     (at_kitchen_content content230)
     (not_allergic_gluten child160)
     (not_allergic_gluten child449)
     (not_allergic_gluten child14)
     (waiting child14 table232)
     (waiting child160 table275)
     (waiting child449 table275)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child14)
     (served child160)
     (served child449)
    )
  )
)
