; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 274342

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child55 - child
    bread43 - bread-portion
    content290 - content-portion
    tray434 tray11 tray322 - tray
    table486 table477 table184 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray434 kitchen)
     (at tray11 kitchen)
     (at tray322 kitchen)
     (at_kitchen_bread bread43)
     (at_kitchen_content content290)
     (not_allergic_gluten child55)
     (waiting child55 table486)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child55)
    )
  )
)
