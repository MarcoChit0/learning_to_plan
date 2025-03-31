; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 367608

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child498 - child
    bread382 - bread-portion
    content65 - content-portion
    tray377 - tray
    table358 table239 table241 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray377 kitchen)
     (at_kitchen_bread bread382)
     (at_kitchen_content content65)
     (not_allergic_gluten child498)
     (waiting child498 table358)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child498)
    )
  )
)
