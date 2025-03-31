; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 838839

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child13 - child
    bread329 - bread-portion
    content260 - content-portion
    tray52 tray304 - tray
    table27 table393 table272 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray52 kitchen)
     (at tray304 kitchen)
     (at_kitchen_bread bread329)
     (at_kitchen_content content260)
     (not_allergic_gluten child13)
     (waiting child13 table272)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child13)
    )
  )
)
