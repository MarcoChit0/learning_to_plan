; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 364886

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child481 - child
    bread278 - bread-portion
    content280 - content-portion
    tray176 tray47 - tray
    table64 table68 table175 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray176 kitchen)
     (at tray47 kitchen)
     (at_kitchen_bread bread278)
     (at_kitchen_content content280)
     (not_allergic_gluten child481)
     (waiting child481 table64)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child481)
    )
  )
)
