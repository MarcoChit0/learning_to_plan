; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 211265

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child407 - child
    bread376 - bread-portion
    content125 - content-portion
    tray108 tray260 tray101 - tray
    table187 table224 table319 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray108 kitchen)
     (at tray260 kitchen)
     (at tray101 kitchen)
     (at_kitchen_bread bread376)
     (at_kitchen_content content125)
     (not_allergic_gluten child407)
     (waiting child407 table187)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child407)
    )
  )
)
