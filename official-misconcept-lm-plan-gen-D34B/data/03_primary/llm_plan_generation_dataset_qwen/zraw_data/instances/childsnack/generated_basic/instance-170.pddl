; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 357556

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child259 child488 - child
    bread454 bread415 - bread-portion
    content475 content112 - content-portion
    tray117 tray316 tray190 tray403 - tray
    table107 table53 table246 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray117 kitchen)
     (at tray316 kitchen)
     (at tray190 kitchen)
     (at tray403 kitchen)
     (at_kitchen_bread bread454)
     (at_kitchen_bread bread415)
     (at_kitchen_content content475)
     (at_kitchen_content content112)
     (not_allergic_gluten child488)
     (not_allergic_gluten child259)
     (waiting child259 table107)
     (waiting child488 table53)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child259)
     (served child488)
    )
  )
)
