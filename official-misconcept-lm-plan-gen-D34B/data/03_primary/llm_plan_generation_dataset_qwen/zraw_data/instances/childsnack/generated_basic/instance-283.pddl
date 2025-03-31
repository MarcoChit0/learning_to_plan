; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 161619

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child254 - child
    bread121 - bread-portion
    content258 - content-portion
    tray252 - tray
    table404 table141 table147 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray252 kitchen)
     (at_kitchen_bread bread121)
     (at_kitchen_content content258)
     (not_allergic_gluten child254)
     (waiting child254 table141)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child254)
    )
  )
)
