; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 63966

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child176 child150 - child
    bread33 bread328 - bread-portion
    content113 content28 - content-portion
    tray200 tray472 - tray
    table452 table436 table16 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at tray472 kitchen)
     (at_kitchen_bread bread33)
     (at_kitchen_bread bread328)
     (at_kitchen_content content113)
     (at_kitchen_content content28)
     (no_gluten_bread bread33)
     (no_gluten_content content113)
     (allergic_gluten child176)
     (not_allergic_gluten child150)
     (waiting child176 table16)
     (waiting child150 table436)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child176)
     (served child150)
    )
  )
)
