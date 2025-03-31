; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 66760

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child465 child328 child12 - child
    bread385 bread54 bread427 - bread-portion
    content1 content405 content487 - content-portion
    tray163 - tray
    table414 table318 table475 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray163 kitchen)
     (at_kitchen_bread bread385)
     (at_kitchen_bread bread54)
     (at_kitchen_bread bread427)
     (at_kitchen_content content1)
     (at_kitchen_content content405)
     (at_kitchen_content content487)
     (not_allergic_gluten child12)
     (not_allergic_gluten child465)
     (not_allergic_gluten child328)
     (waiting child465 table318)
     (waiting child328 table475)
     (waiting child12 table414)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child465)
     (served child328)
     (served child12)
    )
  )
)
