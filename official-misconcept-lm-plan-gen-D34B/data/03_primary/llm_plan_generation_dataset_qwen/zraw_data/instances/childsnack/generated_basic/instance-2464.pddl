; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 578113

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child293 child121 - child
    bread25 bread141 - bread-portion
    content384 content13 - content-portion
    tray497 tray385 - tray
    table212 table448 table263 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray497 kitchen)
     (at tray385 kitchen)
     (at_kitchen_bread bread25)
     (at_kitchen_bread bread141)
     (at_kitchen_content content384)
     (at_kitchen_content content13)
     (not_allergic_gluten child121)
     (not_allergic_gluten child293)
     (waiting child293 table448)
     (waiting child121 table263)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child293)
     (served child121)
    )
  )
)
