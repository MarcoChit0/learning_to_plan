; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 570373

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child222 child11 - child
    bread165 bread1 - bread-portion
    content469 content230 - content-portion
    tray483 tray32 tray485 tray285 - tray
    table115 table20 table13 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at tray32 kitchen)
     (at tray485 kitchen)
     (at tray285 kitchen)
     (at_kitchen_bread bread165)
     (at_kitchen_bread bread1)
     (at_kitchen_content content469)
     (at_kitchen_content content230)
     (not_allergic_gluten child222)
     (not_allergic_gluten child11)
     (waiting child222 table115)
     (waiting child11 table20)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child222)
     (served child11)
    )
  )
)
