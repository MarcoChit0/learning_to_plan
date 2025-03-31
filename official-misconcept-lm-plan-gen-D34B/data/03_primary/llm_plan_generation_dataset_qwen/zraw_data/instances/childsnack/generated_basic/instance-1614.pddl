; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 879242

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child395 - child
    bread82 - bread-portion
    content166 - content-portion
    tray194 tray387 tray489 tray241 - tray
    table13 table294 table208 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray194 kitchen)
     (at tray387 kitchen)
     (at tray489 kitchen)
     (at tray241 kitchen)
     (at_kitchen_bread bread82)
     (at_kitchen_content content166)
     (not_allergic_gluten child395)
     (waiting child395 table208)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child395)
    )
  )
)
