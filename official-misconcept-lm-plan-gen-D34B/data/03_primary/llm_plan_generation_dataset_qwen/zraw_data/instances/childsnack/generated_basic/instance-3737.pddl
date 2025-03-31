; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 373744

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child261 - child
    bread226 - bread-portion
    content135 - content-portion
    tray436 tray222 tray282 - tray
    table49 table285 table213 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray436 kitchen)
     (at tray222 kitchen)
     (at tray282 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_content content135)
     (not_allergic_gluten child261)
     (waiting child261 table49)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child261)
    )
  )
)
