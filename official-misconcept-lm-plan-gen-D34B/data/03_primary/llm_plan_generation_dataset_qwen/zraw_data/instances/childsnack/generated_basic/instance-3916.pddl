; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 876080

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child375 child209 - child
    bread139 bread240 - bread-portion
    content91 content54 - content-portion
    tray179 - tray
    table210 table92 table463 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray179 kitchen)
     (at_kitchen_bread bread139)
     (at_kitchen_bread bread240)
     (at_kitchen_content content91)
     (at_kitchen_content content54)
     (no_gluten_bread bread139)
     (no_gluten_content content91)
     (allergic_gluten child375)
     (not_allergic_gluten child209)
     (waiting child375 table210)
     (waiting child209 table92)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child375)
     (served child209)
    )
  )
)
