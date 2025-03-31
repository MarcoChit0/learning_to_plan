; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 693865

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child116 - child
    bread140 - bread-portion
    content83 - content-portion
    tray191 tray472 tray172 tray356 - tray
    table431 table438 table338 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at tray472 kitchen)
     (at tray172 kitchen)
     (at tray356 kitchen)
     (at_kitchen_bread bread140)
     (at_kitchen_content content83)
     (not_allergic_gluten child116)
     (waiting child116 table431)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child116)
    )
  )
)
