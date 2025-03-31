; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 633756

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 - child
    bread97 - bread-portion
    content289 - content-portion
    tray435 - tray
    table120 table55 table258 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray435 kitchen)
     (at_kitchen_bread bread97)
     (at_kitchen_content content289)
     (not_allergic_gluten child109)
     (waiting child109 table258)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child109)
    )
  )
)
