; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 357547

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child38 - child
    bread132 - bread-portion
    content453 - content-portion
    tray66 - tray
    table51 table364 table338 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray66 kitchen)
     (at_kitchen_bread bread132)
     (at_kitchen_content content453)
     (not_allergic_gluten child38)
     (waiting child38 table51)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child38)
    )
  )
)
