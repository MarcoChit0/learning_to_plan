; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 342265

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child90 - child
    bread426 - bread-portion
    content289 - content-portion
    tray212 - tray
    table317 table185 table5 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray212 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_content content289)
     (not_allergic_gluten child90)
     (waiting child90 table317)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child90)
    )
  )
)
