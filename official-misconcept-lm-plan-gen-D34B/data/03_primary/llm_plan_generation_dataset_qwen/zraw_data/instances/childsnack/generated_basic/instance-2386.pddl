; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 728877

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child72 child454 - child
    bread135 bread39 - bread-portion
    content121 content426 - content-portion
    tray294 - tray
    table409 table467 table364 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray294 kitchen)
     (at_kitchen_bread bread135)
     (at_kitchen_bread bread39)
     (at_kitchen_content content121)
     (at_kitchen_content content426)
     (not_allergic_gluten child454)
     (not_allergic_gluten child72)
     (waiting child72 table409)
     (waiting child454 table364)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child72)
     (served child454)
    )
  )
)
