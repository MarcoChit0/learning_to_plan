; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 434475

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child81 child20 - child
    bread112 bread435 - bread-portion
    content208 content270 - content-portion
    tray337 tray103 - tray
    table58 table273 table198 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray337 kitchen)
     (at tray103 kitchen)
     (at_kitchen_bread bread112)
     (at_kitchen_bread bread435)
     (at_kitchen_content content208)
     (at_kitchen_content content270)
     (not_allergic_gluten child20)
     (not_allergic_gluten child81)
     (waiting child81 table273)
     (waiting child20 table58)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child81)
     (served child20)
    )
  )
)
