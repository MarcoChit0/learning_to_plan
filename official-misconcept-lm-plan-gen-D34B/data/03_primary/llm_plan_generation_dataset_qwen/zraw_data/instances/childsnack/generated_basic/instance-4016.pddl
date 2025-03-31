; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 718328

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child125 child257 child332 - child
    bread160 bread381 bread89 - bread-portion
    content166 content201 content123 - content-portion
    tray409 - tray
    table154 table17 table291 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray409 kitchen)
     (at_kitchen_bread bread160)
     (at_kitchen_bread bread381)
     (at_kitchen_bread bread89)
     (at_kitchen_content content166)
     (at_kitchen_content content201)
     (at_kitchen_content content123)
     (not_allergic_gluten child125)
     (not_allergic_gluten child257)
     (not_allergic_gluten child332)
     (waiting child125 table17)
     (waiting child257 table154)
     (waiting child332 table291)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child125)
     (served child257)
     (served child332)
    )
  )
)
