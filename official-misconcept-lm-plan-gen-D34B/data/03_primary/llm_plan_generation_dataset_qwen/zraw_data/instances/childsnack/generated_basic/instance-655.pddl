; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 246018

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child405 child192 - child
    bread151 bread287 - bread-portion
    content426 content181 - content-portion
    tray457 tray14 - tray
    table458 table497 table48 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray457 kitchen)
     (at tray14 kitchen)
     (at_kitchen_bread bread151)
     (at_kitchen_bread bread287)
     (at_kitchen_content content426)
     (at_kitchen_content content181)
     (no_gluten_bread bread151)
     (no_gluten_content content181)
     (allergic_gluten child192)
     (not_allergic_gluten child405)
     (waiting child405 table48)
     (waiting child192 table458)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child405)
     (served child192)
    )
  )
)
