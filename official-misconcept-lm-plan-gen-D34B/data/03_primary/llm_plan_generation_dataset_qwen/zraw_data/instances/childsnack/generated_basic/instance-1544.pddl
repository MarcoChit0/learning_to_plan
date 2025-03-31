; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 519084

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child432 child319 - child
    bread107 bread226 - bread-portion
    content259 content308 - content-portion
    tray393 tray265 tray347 tray348 - tray
    table413 table115 table174 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray393 kitchen)
     (at tray265 kitchen)
     (at tray347 kitchen)
     (at tray348 kitchen)
     (at_kitchen_bread bread107)
     (at_kitchen_bread bread226)
     (at_kitchen_content content259)
     (at_kitchen_content content308)
     (not_allergic_gluten child432)
     (not_allergic_gluten child319)
     (waiting child432 table413)
     (waiting child319 table174)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child432)
     (served child319)
    )
  )
)
