; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 272168

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child116 - child
    bread394 - bread-portion
    content418 - content-portion
    tray291 - tray
    table50 table443 table26 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray291 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_content content418)
     (not_allergic_gluten child116)
     (waiting child116 table26)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child116)
    )
  )
)
