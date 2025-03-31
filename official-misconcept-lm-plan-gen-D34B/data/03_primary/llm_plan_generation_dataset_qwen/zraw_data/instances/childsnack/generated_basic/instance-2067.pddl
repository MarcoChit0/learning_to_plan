; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 354036

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child176 - child
    bread406 - bread-portion
    content49 - content-portion
    tray52 tray376 tray36 tray140 - tray
    table440 table243 table104 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray52 kitchen)
     (at tray376 kitchen)
     (at tray36 kitchen)
     (at tray140 kitchen)
     (at_kitchen_bread bread406)
     (at_kitchen_content content49)
     (not_allergic_gluten child176)
     (waiting child176 table440)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child176)
    )
  )
)
