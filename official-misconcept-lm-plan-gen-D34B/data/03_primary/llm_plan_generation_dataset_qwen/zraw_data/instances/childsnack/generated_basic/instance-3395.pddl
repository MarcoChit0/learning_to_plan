; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 553754

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child281 child332 child419 - child
    bread60 bread134 bread32 - bread-portion
    content37 content241 content469 - content-portion
    tray292 tray467 tray5 - tray
    table257 table339 table223 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray292 kitchen)
     (at tray467 kitchen)
     (at tray5 kitchen)
     (at_kitchen_bread bread60)
     (at_kitchen_bread bread134)
     (at_kitchen_bread bread32)
     (at_kitchen_content content37)
     (at_kitchen_content content241)
     (at_kitchen_content content469)
     (no_gluten_bread bread32)
     (no_gluten_content content37)
     (allergic_gluten child281)
     (not_allergic_gluten child332)
     (not_allergic_gluten child419)
     (waiting child281 table339)
     (waiting child332 table339)
     (waiting child419 table257)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child281)
     (served child332)
     (served child419)
    )
  )
)
