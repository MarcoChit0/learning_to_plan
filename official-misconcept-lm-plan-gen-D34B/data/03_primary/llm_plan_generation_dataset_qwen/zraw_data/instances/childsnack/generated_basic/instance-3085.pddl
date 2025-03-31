; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 391907

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child353 child53 - child
    bread60 bread65 - bread-portion
    content255 content130 - content-portion
    tray22 tray343 tray249 - tray
    table23 table166 table153 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray22 kitchen)
     (at tray343 kitchen)
     (at tray249 kitchen)
     (at_kitchen_bread bread60)
     (at_kitchen_bread bread65)
     (at_kitchen_content content255)
     (at_kitchen_content content130)
     (not_allergic_gluten child53)
     (not_allergic_gluten child353)
     (waiting child353 table166)
     (waiting child53 table153)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child353)
     (served child53)
    )
  )
)
