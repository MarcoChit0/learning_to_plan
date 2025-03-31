; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 686225

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child356 child154 - child
    bread451 bread298 - bread-portion
    content372 content235 - content-portion
    tray357 - tray
    table268 table4 table353 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray357 kitchen)
     (at_kitchen_bread bread451)
     (at_kitchen_bread bread298)
     (at_kitchen_content content372)
     (at_kitchen_content content235)
     (not_allergic_gluten child154)
     (not_allergic_gluten child356)
     (waiting child356 table353)
     (waiting child154 table353)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child356)
     (served child154)
    )
  )
)
