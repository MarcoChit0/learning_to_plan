; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 589096

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child239 child272 child295 - child
    bread430 bread59 bread215 - bread-portion
    content231 content409 content176 - content-portion
    tray21 - tray
    table86 table206 table406 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray21 kitchen)
     (at_kitchen_bread bread430)
     (at_kitchen_bread bread59)
     (at_kitchen_bread bread215)
     (at_kitchen_content content231)
     (at_kitchen_content content409)
     (at_kitchen_content content176)
     (not_allergic_gluten child272)
     (not_allergic_gluten child239)
     (not_allergic_gluten child295)
     (waiting child239 table206)
     (waiting child272 table206)
     (waiting child295 table406)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child239)
     (served child272)
     (served child295)
    )
  )
)
