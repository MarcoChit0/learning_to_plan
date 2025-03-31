; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 244996

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child411 child223 - child
    bread184 bread208 - bread-portion
    content69 content330 - content-portion
    tray379 tray55 tray228 - tray
    table102 table10 table228 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray379 kitchen)
     (at tray55 kitchen)
     (at tray228 kitchen)
     (at_kitchen_bread bread184)
     (at_kitchen_bread bread208)
     (at_kitchen_content content69)
     (at_kitchen_content content330)
     (no_gluten_bread bread184)
     (no_gluten_content content69)
     (allergic_gluten child411)
     (not_allergic_gluten child223)
     (waiting child411 table10)
     (waiting child223 table10)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child411)
     (served child223)
    )
  )
)
