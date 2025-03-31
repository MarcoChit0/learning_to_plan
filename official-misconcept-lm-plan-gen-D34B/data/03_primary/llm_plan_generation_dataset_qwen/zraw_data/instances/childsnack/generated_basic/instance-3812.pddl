; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 518129

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child463 - child
    bread100 - bread-portion
    content374 - content-portion
    tray225 tray223 tray437 tray439 - tray
    table328 table95 table327 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray225 kitchen)
     (at tray223 kitchen)
     (at tray437 kitchen)
     (at tray439 kitchen)
     (at_kitchen_bread bread100)
     (at_kitchen_content content374)
     (not_allergic_gluten child463)
     (waiting child463 table327)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child463)
    )
  )
)
