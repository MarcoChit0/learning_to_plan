; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 372669

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child100 child86 - child
    bread184 bread172 - bread-portion
    content39 content255 - content-portion
    tray39 tray158 tray117 - tray
    table472 table376 table481 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray39 kitchen)
     (at tray158 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread184)
     (at_kitchen_bread bread172)
     (at_kitchen_content content39)
     (at_kitchen_content content255)
     (not_allergic_gluten child100)
     (not_allergic_gluten child86)
     (waiting child100 table376)
     (waiting child86 table481)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child100)
     (served child86)
    )
  )
)
