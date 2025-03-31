; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 859575

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child333 - child
    bread283 - bread-portion
    content387 - content-portion
    tray277 - tray
    table51 table104 table460 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray277 kitchen)
     (at_kitchen_bread bread283)
     (at_kitchen_content content387)
     (not_allergic_gluten child333)
     (waiting child333 table104)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child333)
    )
  )
)
