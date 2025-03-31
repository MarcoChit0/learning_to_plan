; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 268333

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child120 child281 - child
    bread306 bread184 - bread-portion
    content412 content362 - content-portion
    tray273 tray206 - tray
    table21 table155 table256 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray273 kitchen)
     (at tray206 kitchen)
     (at_kitchen_bread bread306)
     (at_kitchen_bread bread184)
     (at_kitchen_content content412)
     (at_kitchen_content content362)
     (not_allergic_gluten child120)
     (not_allergic_gluten child281)
     (waiting child120 table256)
     (waiting child281 table256)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child120)
     (served child281)
    )
  )
)
