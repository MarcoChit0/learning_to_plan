; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 913580

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child285 - child
    bread317 - bread-portion
    content414 - content-portion
    tray467 tray164 tray168 - tray
    table153 table44 table449 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray467 kitchen)
     (at tray164 kitchen)
     (at tray168 kitchen)
     (at_kitchen_bread bread317)
     (at_kitchen_content content414)
     (not_allergic_gluten child285)
     (waiting child285 table153)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child285)
    )
  )
)
