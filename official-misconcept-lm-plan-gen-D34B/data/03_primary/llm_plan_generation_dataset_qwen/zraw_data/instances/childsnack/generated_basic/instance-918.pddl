; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 347300

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child271 child1 child462 - child
    bread101 bread364 bread95 - bread-portion
    content123 content17 content154 - content-portion
    tray41 - tray
    table75 table242 table24 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray41 kitchen)
     (at_kitchen_bread bread101)
     (at_kitchen_bread bread364)
     (at_kitchen_bread bread95)
     (at_kitchen_content content123)
     (at_kitchen_content content17)
     (at_kitchen_content content154)
     (not_allergic_gluten child271)
     (not_allergic_gluten child1)
     (not_allergic_gluten child462)
     (waiting child271 table242)
     (waiting child1 table24)
     (waiting child462 table75)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child271)
     (served child1)
     (served child462)
    )
  )
)
