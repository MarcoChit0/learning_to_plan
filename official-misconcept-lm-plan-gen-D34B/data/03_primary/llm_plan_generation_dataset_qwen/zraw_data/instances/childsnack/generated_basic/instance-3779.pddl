; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 809718

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child415 - child
    bread230 - bread-portion
    content256 - content-portion
    tray336 - tray
    table26 table273 table156 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray336 kitchen)
     (at_kitchen_bread bread230)
     (at_kitchen_content content256)
     (not_allergic_gluten child415)
     (waiting child415 table273)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child415)
    )
  )
)
