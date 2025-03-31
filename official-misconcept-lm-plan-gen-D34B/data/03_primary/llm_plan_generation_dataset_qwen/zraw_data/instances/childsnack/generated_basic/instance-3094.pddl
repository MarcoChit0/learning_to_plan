; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 783504

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child58 child44 - child
    bread110 bread191 - bread-portion
    content81 content481 - content-portion
    tray207 tray159 tray131 - tray
    table252 table120 table9 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray207 kitchen)
     (at tray159 kitchen)
     (at tray131 kitchen)
     (at_kitchen_bread bread110)
     (at_kitchen_bread bread191)
     (at_kitchen_content content81)
     (at_kitchen_content content481)
     (not_allergic_gluten child58)
     (not_allergic_gluten child44)
     (waiting child58 table252)
     (waiting child44 table120)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child58)
     (served child44)
    )
  )
)
