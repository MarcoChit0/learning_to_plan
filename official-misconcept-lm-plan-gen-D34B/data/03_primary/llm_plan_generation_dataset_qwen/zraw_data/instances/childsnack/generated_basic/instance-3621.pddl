; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 864513

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child340 child126 - child
    bread23 bread71 - bread-portion
    content25 content234 - content-portion
    tray171 tray488 tray14 tray205 - tray
    table234 table42 table442 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray171 kitchen)
     (at tray488 kitchen)
     (at tray14 kitchen)
     (at tray205 kitchen)
     (at_kitchen_bread bread23)
     (at_kitchen_bread bread71)
     (at_kitchen_content content25)
     (at_kitchen_content content234)
     (no_gluten_bread bread71)
     (no_gluten_content content234)
     (allergic_gluten child126)
     (not_allergic_gluten child340)
     (waiting child340 table234)
     (waiting child126 table42)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child340)
     (served child126)
    )
  )
)
