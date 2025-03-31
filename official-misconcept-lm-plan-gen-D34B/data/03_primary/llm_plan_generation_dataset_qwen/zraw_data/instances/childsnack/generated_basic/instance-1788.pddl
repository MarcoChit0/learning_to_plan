; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 593068

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child281 - child
    bread157 - bread-portion
    content172 - content-portion
    tray386 - tray
    table21 table445 table306 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray386 kitchen)
     (at_kitchen_bread bread157)
     (at_kitchen_content content172)
     (not_allergic_gluten child281)
     (waiting child281 table306)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child281)
    )
  )
)
