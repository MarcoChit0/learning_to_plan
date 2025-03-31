; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 220961

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child38 child384 - child
    bread23 bread407 - bread-portion
    content473 content107 - content-portion
    tray439 tray156 - tray
    table210 table438 table276 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray439 kitchen)
     (at tray156 kitchen)
     (at_kitchen_bread bread23)
     (at_kitchen_bread bread407)
     (at_kitchen_content content473)
     (at_kitchen_content content107)
     (no_gluten_bread bread23)
     (no_gluten_content content473)
     (allergic_gluten child384)
     (not_allergic_gluten child38)
     (waiting child38 table210)
     (waiting child384 table276)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child38)
     (served child384)
    )
  )
)
