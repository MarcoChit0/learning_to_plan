; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 956820

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child283 child358 child29 - child
    bread14 bread24 bread371 - bread-portion
    content474 content39 content297 - content-portion
    tray193 - tray
    table408 table397 table437 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray193 kitchen)
     (at_kitchen_bread bread14)
     (at_kitchen_bread bread24)
     (at_kitchen_bread bread371)
     (at_kitchen_content content474)
     (at_kitchen_content content39)
     (at_kitchen_content content297)
     (not_allergic_gluten child283)
     (not_allergic_gluten child358)
     (not_allergic_gluten child29)
     (waiting child283 table408)
     (waiting child358 table408)
     (waiting child29 table437)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child283)
     (served child358)
     (served child29)
    )
  )
)
