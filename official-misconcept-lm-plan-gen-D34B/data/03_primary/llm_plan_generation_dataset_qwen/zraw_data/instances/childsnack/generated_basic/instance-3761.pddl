; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 762869

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child306 - child
    bread46 - bread-portion
    content316 - content-portion
    tray400 - tray
    table148 table146 table323 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray400 kitchen)
     (at_kitchen_bread bread46)
     (at_kitchen_content content316)
     (not_allergic_gluten child306)
     (waiting child306 table323)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child306)
    )
  )
)
