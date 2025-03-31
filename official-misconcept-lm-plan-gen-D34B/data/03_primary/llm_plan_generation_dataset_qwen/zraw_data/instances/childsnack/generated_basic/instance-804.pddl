; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 131432

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child497 - child
    bread314 - bread-portion
    content270 - content-portion
    tray68 tray497 - tray
    table199 table299 table409 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray68 kitchen)
     (at tray497 kitchen)
     (at_kitchen_bread bread314)
     (at_kitchen_content content270)
     (not_allergic_gluten child497)
     (waiting child497 table199)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child497)
    )
  )
)
