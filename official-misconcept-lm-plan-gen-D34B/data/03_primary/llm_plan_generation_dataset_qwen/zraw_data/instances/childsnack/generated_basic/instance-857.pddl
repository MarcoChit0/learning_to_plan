; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 972303

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child10 child144 - child
    bread77 bread44 - bread-portion
    content57 content243 - content-portion
    tray19 tray239 - tray
    table238 table404 table256 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray19 kitchen)
     (at tray239 kitchen)
     (at_kitchen_bread bread77)
     (at_kitchen_bread bread44)
     (at_kitchen_content content57)
     (at_kitchen_content content243)
     (no_gluten_bread bread44)
     (no_gluten_content content243)
     (allergic_gluten child144)
     (not_allergic_gluten child10)
     (waiting child10 table256)
     (waiting child144 table256)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child10)
     (served child144)
    )
  )
)
