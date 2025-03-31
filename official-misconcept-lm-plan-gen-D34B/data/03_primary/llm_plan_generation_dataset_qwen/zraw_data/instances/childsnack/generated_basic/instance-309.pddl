; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 213776

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child488 - child
    bread282 - bread-portion
    content232 - content-portion
    tray364 - tray
    table169 table468 table162 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray364 kitchen)
     (at_kitchen_bread bread282)
     (at_kitchen_content content232)
     (not_allergic_gluten child488)
     (waiting child488 table169)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child488)
    )
  )
)
