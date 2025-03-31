; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 294414

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child300 - child
    bread189 - bread-portion
    content331 - content-portion
    tray489 - tray
    table173 table255 table23 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray489 kitchen)
     (at_kitchen_bread bread189)
     (at_kitchen_content content331)
     (not_allergic_gluten child300)
     (waiting child300 table23)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child300)
    )
  )
)
