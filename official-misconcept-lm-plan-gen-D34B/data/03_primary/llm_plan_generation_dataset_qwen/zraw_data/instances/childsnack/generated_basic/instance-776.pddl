; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 865962

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child497 child336 - child
    bread296 bread42 - bread-portion
    content263 content288 - content-portion
    tray40 tray362 - tray
    table64 table165 table354 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray40 kitchen)
     (at tray362 kitchen)
     (at_kitchen_bread bread296)
     (at_kitchen_bread bread42)
     (at_kitchen_content content263)
     (at_kitchen_content content288)
     (no_gluten_bread bread42)
     (no_gluten_content content263)
     (allergic_gluten child336)
     (not_allergic_gluten child497)
     (waiting child497 table354)
     (waiting child336 table64)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child497)
     (served child336)
    )
  )
)
