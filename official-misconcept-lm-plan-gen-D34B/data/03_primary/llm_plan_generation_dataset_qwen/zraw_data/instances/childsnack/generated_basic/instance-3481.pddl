; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 10436

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child302 child438 - child
    bread147 bread328 - bread-portion
    content260 content281 - content-portion
    tray288 tray367 - tray
    table231 table263 table300 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray288 kitchen)
     (at tray367 kitchen)
     (at_kitchen_bread bread147)
     (at_kitchen_bread bread328)
     (at_kitchen_content content260)
     (at_kitchen_content content281)
     (no_gluten_bread bread328)
     (no_gluten_content content281)
     (allergic_gluten child438)
     (not_allergic_gluten child302)
     (waiting child302 table263)
     (waiting child438 table300)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child302)
     (served child438)
    )
  )
)
