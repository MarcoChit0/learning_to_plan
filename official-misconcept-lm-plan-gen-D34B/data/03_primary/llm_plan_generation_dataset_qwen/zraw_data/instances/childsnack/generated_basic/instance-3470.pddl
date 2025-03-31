; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 280116

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child328 child498 - child
    bread100 bread326 - bread-portion
    content137 content4 - content-portion
    tray191 tray450 - tray
    table336 table46 table157 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at tray450 kitchen)
     (at_kitchen_bread bread100)
     (at_kitchen_bread bread326)
     (at_kitchen_content content137)
     (at_kitchen_content content4)
     (no_gluten_bread bread100)
     (no_gluten_content content137)
     (allergic_gluten child498)
     (not_allergic_gluten child328)
     (waiting child328 table157)
     (waiting child498 table157)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child328)
     (served child498)
    )
  )
)
