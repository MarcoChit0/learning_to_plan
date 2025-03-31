; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 70871

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child486 - child
    bread302 - bread-portion
    content272 - content-portion
    tray191 - tray
    table428 table326 table203 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at_kitchen_bread bread302)
     (at_kitchen_content content272)
     (not_allergic_gluten child486)
     (waiting child486 table428)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child486)
    )
  )
)
