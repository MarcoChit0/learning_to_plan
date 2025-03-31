; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 594596

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child212 - child
    bread160 - bread-portion
    content76 - content-portion
    tray91 - tray
    table99 table336 table397 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray91 kitchen)
     (at_kitchen_bread bread160)
     (at_kitchen_content content76)
     (not_allergic_gluten child212)
     (waiting child212 table397)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child212)
    )
  )
)
