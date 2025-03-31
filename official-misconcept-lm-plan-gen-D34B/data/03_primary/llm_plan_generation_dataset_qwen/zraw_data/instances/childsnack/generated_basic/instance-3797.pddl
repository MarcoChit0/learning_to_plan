; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 887682

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child187 - child
    bread293 - bread-portion
    content271 - content-portion
    tray420 - tray
    table123 table49 table288 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray420 kitchen)
     (at_kitchen_bread bread293)
     (at_kitchen_content content271)
     (not_allergic_gluten child187)
     (waiting child187 table288)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child187)
    )
  )
)
