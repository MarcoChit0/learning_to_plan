; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 761809

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child347 child377 - child
    bread189 bread102 - bread-portion
    content476 content141 - content-portion
    tray110 tray49 tray162 - tray
    table160 table130 table228 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray110 kitchen)
     (at tray49 kitchen)
     (at tray162 kitchen)
     (at_kitchen_bread bread189)
     (at_kitchen_bread bread102)
     (at_kitchen_content content476)
     (at_kitchen_content content141)
     (not_allergic_gluten child347)
     (not_allergic_gluten child377)
     (waiting child347 table130)
     (waiting child377 table160)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child347)
     (served child377)
    )
  )
)
