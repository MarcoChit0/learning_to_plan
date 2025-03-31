; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 488523

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child370 child210 - child
    bread157 bread432 - bread-portion
    content361 content469 - content-portion
    tray67 tray347 tray389 - tray
    table320 table260 table183 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray67 kitchen)
     (at tray347 kitchen)
     (at tray389 kitchen)
     (at_kitchen_bread bread157)
     (at_kitchen_bread bread432)
     (at_kitchen_content content361)
     (at_kitchen_content content469)
     (not_allergic_gluten child370)
     (not_allergic_gluten child210)
     (waiting child370 table260)
     (waiting child210 table183)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child370)
     (served child210)
    )
  )
)
