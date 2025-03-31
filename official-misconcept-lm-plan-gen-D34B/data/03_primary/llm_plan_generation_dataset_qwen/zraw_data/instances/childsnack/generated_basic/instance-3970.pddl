; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 685861

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child29 - child
    bread6 - bread-portion
    content485 - content-portion
    tray243 tray86 tray484 - tray
    table133 table31 table486 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray243 kitchen)
     (at tray86 kitchen)
     (at tray484 kitchen)
     (at_kitchen_bread bread6)
     (at_kitchen_content content485)
     (not_allergic_gluten child29)
     (waiting child29 table486)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child29)
    )
  )
)
