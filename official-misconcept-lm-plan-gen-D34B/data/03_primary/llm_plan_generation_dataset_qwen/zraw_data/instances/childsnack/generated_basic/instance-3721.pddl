; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 493671

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child203 - child
    bread302 - bread-portion
    content273 - content-portion
    tray481 tray84 tray356 - tray
    table242 table127 table361 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray481 kitchen)
     (at tray84 kitchen)
     (at tray356 kitchen)
     (at_kitchen_bread bread302)
     (at_kitchen_content content273)
     (not_allergic_gluten child203)
     (waiting child203 table127)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child203)
    )
  )
)
