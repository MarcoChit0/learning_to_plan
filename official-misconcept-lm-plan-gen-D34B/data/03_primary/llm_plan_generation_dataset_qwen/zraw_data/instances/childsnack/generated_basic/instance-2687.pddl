; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 579041

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child424 child239 child166 - child
    bread362 bread201 bread111 - bread-portion
    content26 content481 content367 - content-portion
    tray64 tray97 tray77 - tray
    table472 table9 table242 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray64 kitchen)
     (at tray97 kitchen)
     (at tray77 kitchen)
     (at_kitchen_bread bread362)
     (at_kitchen_bread bread201)
     (at_kitchen_bread bread111)
     (at_kitchen_content content26)
     (at_kitchen_content content481)
     (at_kitchen_content content367)
     (not_allergic_gluten child424)
     (not_allergic_gluten child239)
     (not_allergic_gluten child166)
     (waiting child424 table472)
     (waiting child239 table242)
     (waiting child166 table472)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child424)
     (served child239)
     (served child166)
    )
  )
)
