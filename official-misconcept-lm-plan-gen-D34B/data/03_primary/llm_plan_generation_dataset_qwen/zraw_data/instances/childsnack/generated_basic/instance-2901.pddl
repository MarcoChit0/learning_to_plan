; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 224794

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child359 child81 - child
    bread419 bread122 - bread-portion
    content61 content19 - content-portion
    tray237 tray230 tray393 - tray
    table241 table236 table62 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray237 kitchen)
     (at tray230 kitchen)
     (at tray393 kitchen)
     (at_kitchen_bread bread419)
     (at_kitchen_bread bread122)
     (at_kitchen_content content61)
     (at_kitchen_content content19)
     (no_gluten_bread bread122)
     (no_gluten_content content19)
     (allergic_gluten child81)
     (not_allergic_gluten child359)
     (waiting child359 table62)
     (waiting child81 table62)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child359)
     (served child81)
    )
  )
)
