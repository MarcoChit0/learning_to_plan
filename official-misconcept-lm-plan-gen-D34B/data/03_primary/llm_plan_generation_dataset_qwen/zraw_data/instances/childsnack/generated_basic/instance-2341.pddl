; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 889699

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child160 - child
    bread413 - bread-portion
    content182 - content-portion
    tray476 - tray
    table332 table168 table296 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray476 kitchen)
     (at_kitchen_bread bread413)
     (at_kitchen_content content182)
     (not_allergic_gluten child160)
     (waiting child160 table332)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child160)
    )
  )
)
