; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 815484

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child13 child215 - child
    bread111 bread497 - bread-portion
    content270 content17 - content-portion
    tray304 tray67 - tray
    table378 table440 table101 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray304 kitchen)
     (at tray67 kitchen)
     (at_kitchen_bread bread111)
     (at_kitchen_bread bread497)
     (at_kitchen_content content270)
     (at_kitchen_content content17)
     (no_gluten_bread bread497)
     (no_gluten_content content17)
     (allergic_gluten child215)
     (not_allergic_gluten child13)
     (waiting child13 table101)
     (waiting child215 table440)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child13)
     (served child215)
    )
  )
)
