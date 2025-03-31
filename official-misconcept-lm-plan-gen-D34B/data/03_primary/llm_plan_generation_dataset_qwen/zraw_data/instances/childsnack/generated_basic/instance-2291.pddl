; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 913960

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child338 child125 - child
    bread380 bread15 - bread-portion
    content89 content431 - content-portion
    tray200 - tray
    table469 table253 table465 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at_kitchen_bread bread380)
     (at_kitchen_bread bread15)
     (at_kitchen_content content89)
     (at_kitchen_content content431)
     (no_gluten_bread bread15)
     (no_gluten_content content89)
     (allergic_gluten child338)
     (not_allergic_gluten child125)
     (waiting child338 table465)
     (waiting child125 table465)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child338)
     (served child125)
    )
  )
)
