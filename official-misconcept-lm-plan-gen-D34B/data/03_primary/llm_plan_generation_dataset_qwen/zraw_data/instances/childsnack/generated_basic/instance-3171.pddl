; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 224690

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child462 child125 - child
    bread203 bread402 - bread-portion
    content128 content380 - content-portion
    tray322 tray9 - tray
    table223 table128 table485 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray322 kitchen)
     (at tray9 kitchen)
     (at_kitchen_bread bread203)
     (at_kitchen_bread bread402)
     (at_kitchen_content content128)
     (at_kitchen_content content380)
     (not_allergic_gluten child462)
     (not_allergic_gluten child125)
     (waiting child462 table223)
     (waiting child125 table485)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child462)
     (served child125)
    )
  )
)
