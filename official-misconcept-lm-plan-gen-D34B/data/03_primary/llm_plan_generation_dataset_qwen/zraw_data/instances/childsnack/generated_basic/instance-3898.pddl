; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 869384

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child17 - child
    bread22 - bread-portion
    content117 - content-portion
    tray305 tray212 - tray
    table91 table397 table442 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray305 kitchen)
     (at tray212 kitchen)
     (at_kitchen_bread bread22)
     (at_kitchen_content content117)
     (not_allergic_gluten child17)
     (waiting child17 table397)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child17)
    )
  )
)
