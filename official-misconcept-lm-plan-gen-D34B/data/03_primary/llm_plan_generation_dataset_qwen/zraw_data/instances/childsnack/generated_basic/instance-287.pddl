; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 426918

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child298 - child
    bread271 - bread-portion
    content454 - content-portion
    tray308 - tray
    table108 table32 table281 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray308 kitchen)
     (at_kitchen_bread bread271)
     (at_kitchen_content content454)
     (not_allergic_gluten child298)
     (waiting child298 table281)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child298)
    )
  )
)
