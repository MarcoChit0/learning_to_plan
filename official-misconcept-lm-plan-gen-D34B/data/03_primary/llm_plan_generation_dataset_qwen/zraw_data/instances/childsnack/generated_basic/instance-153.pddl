; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 358469

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child22 child9 - child
    bread212 bread421 - bread-portion
    content303 content91 - content-portion
    tray437 tray375 tray162 tray423 - tray
    table318 table209 table483 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray437 kitchen)
     (at tray375 kitchen)
     (at tray162 kitchen)
     (at tray423 kitchen)
     (at_kitchen_bread bread212)
     (at_kitchen_bread bread421)
     (at_kitchen_content content303)
     (at_kitchen_content content91)
     (not_allergic_gluten child9)
     (not_allergic_gluten child22)
     (waiting child22 table318)
     (waiting child9 table483)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child22)
     (served child9)
    )
  )
)
