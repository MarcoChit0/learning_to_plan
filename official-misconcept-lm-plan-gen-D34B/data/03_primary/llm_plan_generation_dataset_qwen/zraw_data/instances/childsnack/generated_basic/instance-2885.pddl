; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 596738

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child295 - child
    bread498 - bread-portion
    content2 - content-portion
    tray486 tray455 - tray
    table129 table40 table284 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray486 kitchen)
     (at tray455 kitchen)
     (at_kitchen_bread bread498)
     (at_kitchen_content content2)
     (not_allergic_gluten child295)
     (waiting child295 table40)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child295)
    )
  )
)
