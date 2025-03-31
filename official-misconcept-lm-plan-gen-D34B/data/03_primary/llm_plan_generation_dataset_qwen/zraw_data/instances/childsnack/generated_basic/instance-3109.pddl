; child-snack task with 2 children and 0.56 gluten factor 
; constant factor of 1.3
; random seed: 986214

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child177 child229 - child
    bread234 bread196 - bread-portion
    content119 content242 - content-portion
    tray123 tray272 - tray
    table290 table243 table487 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray123 kitchen)
     (at tray272 kitchen)
     (at_kitchen_bread bread234)
     (at_kitchen_bread bread196)
     (at_kitchen_content content119)
     (at_kitchen_content content242)
     (no_gluten_bread bread234)
     (no_gluten_content content242)
     (allergic_gluten child229)
     (not_allergic_gluten child177)
     (waiting child177 table243)
     (waiting child229 table290)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child177)
     (served child229)
    )
  )
)
