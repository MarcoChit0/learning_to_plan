; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 734619

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child469 - child
    bread293 - bread-portion
    content469 - content-portion
    tray267 tray260 tray475 tray299 - tray
    table240 table122 table152 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray267 kitchen)
     (at tray260 kitchen)
     (at tray475 kitchen)
     (at tray299 kitchen)
     (at_kitchen_bread bread293)
     (at_kitchen_content content469)
     (not_allergic_gluten child469)
     (waiting child469 table122)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child469)
    )
  )
)
