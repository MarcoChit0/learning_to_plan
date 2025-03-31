; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 498433

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child268 child16 - child
    bread387 bread219 - bread-portion
    content388 content91 - content-portion
    tray39 tray106 - tray
    table127 table189 table19 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray39 kitchen)
     (at tray106 kitchen)
     (at_kitchen_bread bread387)
     (at_kitchen_bread bread219)
     (at_kitchen_content content388)
     (at_kitchen_content content91)
     (no_gluten_bread bread219)
     (no_gluten_content content91)
     (allergic_gluten child268)
     (not_allergic_gluten child16)
     (waiting child268 table189)
     (waiting child16 table127)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child268)
     (served child16)
    )
  )
)
