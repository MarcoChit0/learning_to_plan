; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 668026

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child328 child470 - child
    bread356 bread328 - bread-portion
    content427 content224 - content-portion
    tray91 tray147 - tray
    table48 table63 table498 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray91 kitchen)
     (at tray147 kitchen)
     (at_kitchen_bread bread356)
     (at_kitchen_bread bread328)
     (at_kitchen_content content427)
     (at_kitchen_content content224)
     (not_allergic_gluten child328)
     (not_allergic_gluten child470)
     (waiting child328 table48)
     (waiting child470 table498)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child328)
     (served child470)
    )
  )
)
