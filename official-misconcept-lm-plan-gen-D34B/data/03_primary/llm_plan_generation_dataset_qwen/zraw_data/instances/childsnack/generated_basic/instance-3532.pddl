; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 462369

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child353 child295 - child
    bread205 bread428 - bread-portion
    content106 content128 - content-portion
    tray182 tray329 - tray
    table1 table61 table195 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray182 kitchen)
     (at tray329 kitchen)
     (at_kitchen_bread bread205)
     (at_kitchen_bread bread428)
     (at_kitchen_content content106)
     (at_kitchen_content content128)
     (not_allergic_gluten child295)
     (not_allergic_gluten child353)
     (waiting child353 table61)
     (waiting child295 table61)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child353)
     (served child295)
    )
  )
)
