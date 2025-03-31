; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 620858

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child487 child464 - child
    bread163 bread376 - bread-portion
    content16 content84 - content-portion
    tray408 tray265 - tray
    table357 table179 table46 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray408 kitchen)
     (at tray265 kitchen)
     (at_kitchen_bread bread163)
     (at_kitchen_bread bread376)
     (at_kitchen_content content16)
     (at_kitchen_content content84)
     (not_allergic_gluten child487)
     (not_allergic_gluten child464)
     (waiting child487 table357)
     (waiting child464 table179)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child487)
     (served child464)
    )
  )
)
