; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 891838

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child426 child135 - child
    bread379 bread295 - bread-portion
    content294 content22 - content-portion
    tray206 tray390 tray483 - tray
    table431 table87 table83 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray206 kitchen)
     (at tray390 kitchen)
     (at tray483 kitchen)
     (at_kitchen_bread bread379)
     (at_kitchen_bread bread295)
     (at_kitchen_content content294)
     (at_kitchen_content content22)
     (not_allergic_gluten child426)
     (not_allergic_gluten child135)
     (waiting child426 table83)
     (waiting child135 table87)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child426)
     (served child135)
    )
  )
)
