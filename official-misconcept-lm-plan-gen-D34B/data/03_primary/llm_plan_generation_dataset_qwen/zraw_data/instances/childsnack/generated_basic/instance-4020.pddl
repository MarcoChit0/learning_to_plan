; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 294987

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child363 child238 child92 - child
    bread442 bread110 bread399 - bread-portion
    content181 content164 content282 - content-portion
    tray304 - tray
    table417 table58 table105 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray304 kitchen)
     (at_kitchen_bread bread442)
     (at_kitchen_bread bread110)
     (at_kitchen_bread bread399)
     (at_kitchen_content content181)
     (at_kitchen_content content164)
     (at_kitchen_content content282)
     (not_allergic_gluten child238)
     (not_allergic_gluten child92)
     (not_allergic_gluten child363)
     (waiting child363 table105)
     (waiting child238 table105)
     (waiting child92 table58)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child363)
     (served child238)
     (served child92)
    )
  )
)
