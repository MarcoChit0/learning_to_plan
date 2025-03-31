; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 406128

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child455 - child
    bread4 - bread-portion
    content405 - content-portion
    tray186 tray464 tray205 - tray
    table13 table331 table477 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray186 kitchen)
     (at tray464 kitchen)
     (at tray205 kitchen)
     (at_kitchen_bread bread4)
     (at_kitchen_content content405)
     (not_allergic_gluten child455)
     (waiting child455 table13)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child455)
    )
  )
)
