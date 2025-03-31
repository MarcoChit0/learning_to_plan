; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 591039

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child469 - child
    bread68 - bread-portion
    content38 - content-portion
    tray295 tray333 tray197 tray409 - tray
    table297 table373 table106 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray295 kitchen)
     (at tray333 kitchen)
     (at tray197 kitchen)
     (at tray409 kitchen)
     (at_kitchen_bread bread68)
     (at_kitchen_content content38)
     (not_allergic_gluten child469)
     (waiting child469 table373)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child469)
    )
  )
)
