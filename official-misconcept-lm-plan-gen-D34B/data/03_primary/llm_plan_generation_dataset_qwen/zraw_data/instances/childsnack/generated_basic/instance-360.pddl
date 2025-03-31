; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 457364

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child173 - child
    bread299 - bread-portion
    content240 - content-portion
    tray179 tray439 tray32 tray176 - tray
    table234 table488 table146 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray179 kitchen)
     (at tray439 kitchen)
     (at tray32 kitchen)
     (at tray176 kitchen)
     (at_kitchen_bread bread299)
     (at_kitchen_content content240)
     (not_allergic_gluten child173)
     (waiting child173 table146)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child173)
    )
  )
)
