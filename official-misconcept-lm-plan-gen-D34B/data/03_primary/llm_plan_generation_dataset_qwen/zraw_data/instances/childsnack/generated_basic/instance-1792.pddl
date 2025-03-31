; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 621824

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child43 - child
    bread117 - bread-portion
    content62 - content-portion
    tray299 - tray
    table302 table118 table188 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray299 kitchen)
     (at_kitchen_bread bread117)
     (at_kitchen_content content62)
     (not_allergic_gluten child43)
     (waiting child43 table302)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child43)
    )
  )
)
