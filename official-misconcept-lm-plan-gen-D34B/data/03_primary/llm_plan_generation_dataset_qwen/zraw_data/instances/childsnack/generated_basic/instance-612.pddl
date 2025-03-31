; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 631910

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child434 - child
    bread56 - bread-portion
    content138 - content-portion
    tray293 - tray
    table453 table58 table420 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray293 kitchen)
     (at_kitchen_bread bread56)
     (at_kitchen_content content138)
     (not_allergic_gluten child434)
     (waiting child434 table453)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child434)
    )
  )
)
