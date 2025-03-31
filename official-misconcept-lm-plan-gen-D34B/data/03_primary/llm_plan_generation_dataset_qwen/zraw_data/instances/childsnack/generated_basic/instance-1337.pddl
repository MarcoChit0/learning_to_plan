; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 514223

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child287 child90 - child
    bread240 bread463 - bread-portion
    content61 content320 - content-portion
    tray219 tray292 tray39 - tray
    table386 table498 table478 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray219 kitchen)
     (at tray292 kitchen)
     (at tray39 kitchen)
     (at_kitchen_bread bread240)
     (at_kitchen_bread bread463)
     (at_kitchen_content content61)
     (at_kitchen_content content320)
     (not_allergic_gluten child90)
     (not_allergic_gluten child287)
     (waiting child287 table386)
     (waiting child90 table386)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child287)
     (served child90)
    )
  )
)
