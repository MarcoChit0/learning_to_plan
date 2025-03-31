; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 998759

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child285 child77 - child
    bread452 bread243 - bread-portion
    content424 content474 - content-portion
    tray394 - tray
    table71 table342 table395 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray394 kitchen)
     (at_kitchen_bread bread452)
     (at_kitchen_bread bread243)
     (at_kitchen_content content424)
     (at_kitchen_content content474)
     (not_allergic_gluten child285)
     (not_allergic_gluten child77)
     (waiting child285 table395)
     (waiting child77 table342)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child285)
     (served child77)
    )
  )
)
