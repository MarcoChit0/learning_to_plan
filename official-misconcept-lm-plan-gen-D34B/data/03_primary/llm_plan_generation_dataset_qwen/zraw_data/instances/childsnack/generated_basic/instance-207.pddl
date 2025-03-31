; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 605923

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child187 child427 - child
    bread376 bread240 - bread-portion
    content384 content128 - content-portion
    tray291 tray462 tray495 - tray
    table319 table408 table64 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray291 kitchen)
     (at tray462 kitchen)
     (at tray495 kitchen)
     (at_kitchen_bread bread376)
     (at_kitchen_bread bread240)
     (at_kitchen_content content384)
     (at_kitchen_content content128)
     (not_allergic_gluten child427)
     (not_allergic_gluten child187)
     (waiting child187 table64)
     (waiting child427 table64)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child187)
     (served child427)
    )
  )
)
