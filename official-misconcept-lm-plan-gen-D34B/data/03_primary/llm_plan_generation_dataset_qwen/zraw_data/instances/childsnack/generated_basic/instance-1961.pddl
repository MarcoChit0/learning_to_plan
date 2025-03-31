; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 106703

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child187 child393 - child
    bread267 bread255 - bread-portion
    content430 content173 - content-portion
    tray429 - tray
    table234 table415 table438 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray429 kitchen)
     (at_kitchen_bread bread267)
     (at_kitchen_bread bread255)
     (at_kitchen_content content430)
     (at_kitchen_content content173)
     (no_gluten_bread bread255)
     (no_gluten_content content173)
     (allergic_gluten child187)
     (not_allergic_gluten child393)
     (waiting child187 table438)
     (waiting child393 table234)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child187)
     (served child393)
    )
  )
)
