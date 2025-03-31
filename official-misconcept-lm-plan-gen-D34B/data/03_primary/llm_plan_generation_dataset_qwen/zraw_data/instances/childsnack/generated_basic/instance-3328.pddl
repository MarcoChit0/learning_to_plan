; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 632999

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child122 child407 - child
    bread363 bread419 - bread-portion
    content409 content310 - content-portion
    tray302 tray338 - tray
    table110 table13 table118 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at tray338 kitchen)
     (at_kitchen_bread bread363)
     (at_kitchen_bread bread419)
     (at_kitchen_content content409)
     (at_kitchen_content content310)
     (not_allergic_gluten child407)
     (not_allergic_gluten child122)
     (waiting child122 table110)
     (waiting child407 table118)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child122)
     (served child407)
    )
  )
)
