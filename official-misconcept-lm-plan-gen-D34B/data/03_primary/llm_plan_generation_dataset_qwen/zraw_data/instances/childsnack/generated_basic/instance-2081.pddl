; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 338458

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child135 - child
    bread65 - bread-portion
    content374 - content-portion
    tray293 tray158 tray189 tray101 - tray
    table408 table451 table61 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray293 kitchen)
     (at tray158 kitchen)
     (at tray189 kitchen)
     (at tray101 kitchen)
     (at_kitchen_bread bread65)
     (at_kitchen_content content374)
     (not_allergic_gluten child135)
     (waiting child135 table408)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child135)
    )
  )
)
