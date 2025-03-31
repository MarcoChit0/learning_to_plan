; child-snack task with 2 children and 0.67 gluten factor 
; constant factor of 1.5
; random seed: 452344

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child111 child422 - child
    bread373 bread416 - bread-portion
    content42 content435 - content-portion
    tray15 - tray
    table455 table456 table27 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray15 kitchen)
     (at_kitchen_bread bread373)
     (at_kitchen_bread bread416)
     (at_kitchen_content content42)
     (at_kitchen_content content435)
     (no_gluten_bread bread373)
     (no_gluten_content content42)
     (allergic_gluten child111)
     (not_allergic_gluten child422)
     (waiting child111 table455)
     (waiting child422 table27)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child111)
     (served child422)
    )
  )
)
