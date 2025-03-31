; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 907337

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child294 - child
    bread235 - bread-portion
    content367 - content-portion
    tray80 - tray
    table486 table364 table27 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray80 kitchen)
     (at_kitchen_bread bread235)
     (at_kitchen_content content367)
     (not_allergic_gluten child294)
     (waiting child294 table486)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child294)
    )
  )
)
