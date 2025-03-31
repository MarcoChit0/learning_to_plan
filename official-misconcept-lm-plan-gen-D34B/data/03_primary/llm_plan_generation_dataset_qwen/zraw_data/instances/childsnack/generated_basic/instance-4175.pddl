; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 931732

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child179 child73 - child
    bread442 bread323 - bread-portion
    content168 content263 - content-portion
    tray339 - tray
    table256 table233 table263 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray339 kitchen)
     (at_kitchen_bread bread442)
     (at_kitchen_bread bread323)
     (at_kitchen_content content168)
     (at_kitchen_content content263)
     (not_allergic_gluten child73)
     (not_allergic_gluten child179)
     (waiting child179 table263)
     (waiting child73 table256)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child179)
     (served child73)
    )
  )
)
