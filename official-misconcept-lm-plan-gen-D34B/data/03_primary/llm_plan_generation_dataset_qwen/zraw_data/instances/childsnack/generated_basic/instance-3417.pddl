; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 925743

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child108 - child
    bread108 - bread-portion
    content341 - content-portion
    tray244 tray241 tray139 tray489 - tray
    table20 table255 table103 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray244 kitchen)
     (at tray241 kitchen)
     (at tray139 kitchen)
     (at tray489 kitchen)
     (at_kitchen_bread bread108)
     (at_kitchen_content content341)
     (not_allergic_gluten child108)
     (waiting child108 table20)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child108)
    )
  )
)
