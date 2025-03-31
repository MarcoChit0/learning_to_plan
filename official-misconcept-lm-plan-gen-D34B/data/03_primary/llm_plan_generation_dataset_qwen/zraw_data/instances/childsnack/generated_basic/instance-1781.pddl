; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 526978

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child246 - child
    bread155 - bread-portion
    content484 - content-portion
    tray163 - tray
    table45 table92 table70 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray163 kitchen)
     (at_kitchen_bread bread155)
     (at_kitchen_content content484)
     (not_allergic_gluten child246)
     (waiting child246 table92)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child246)
    )
  )
)
