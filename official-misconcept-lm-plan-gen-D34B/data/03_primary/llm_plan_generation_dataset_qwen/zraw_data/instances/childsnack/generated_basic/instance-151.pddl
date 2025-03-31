; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 338378

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child479 child252 - child
    bread473 bread372 - bread-portion
    content220 content35 - content-portion
    tray238 tray50 tray448 tray68 - tray
    table326 table34 table211 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray238 kitchen)
     (at tray50 kitchen)
     (at tray448 kitchen)
     (at tray68 kitchen)
     (at_kitchen_bread bread473)
     (at_kitchen_bread bread372)
     (at_kitchen_content content220)
     (at_kitchen_content content35)
     (not_allergic_gluten child252)
     (not_allergic_gluten child479)
     (waiting child479 table34)
     (waiting child252 table326)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child479)
     (served child252)
    )
  )
)
