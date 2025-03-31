; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 716101

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child140 - child
    bread51 - bread-portion
    content484 - content-portion
    tray423 - tray
    table82 table116 table3 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray423 kitchen)
     (at_kitchen_bread bread51)
     (at_kitchen_content content484)
     (not_allergic_gluten child140)
     (waiting child140 table82)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child140)
    )
  )
)
