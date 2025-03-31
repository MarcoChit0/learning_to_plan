; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 757161

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child10 child213 - child
    bread229 bread106 - bread-portion
    content180 content36 - content-portion
    tray382 tray270 - tray
    table199 table45 table410 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray382 kitchen)
     (at tray270 kitchen)
     (at_kitchen_bread bread229)
     (at_kitchen_bread bread106)
     (at_kitchen_content content180)
     (at_kitchen_content content36)
     (no_gluten_bread bread229)
     (no_gluten_content content180)
     (allergic_gluten child213)
     (not_allergic_gluten child10)
     (waiting child10 table410)
     (waiting child213 table45)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child10)
     (served child213)
    )
  )
)
