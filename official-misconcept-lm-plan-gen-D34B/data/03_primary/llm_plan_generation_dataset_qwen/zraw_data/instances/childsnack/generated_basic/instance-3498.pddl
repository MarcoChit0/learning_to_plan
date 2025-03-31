; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 900903

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child395 child239 - child
    bread277 bread22 - bread-portion
    content410 content137 - content-portion
    tray182 tray99 - tray
    table234 table329 table379 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray182 kitchen)
     (at tray99 kitchen)
     (at_kitchen_bread bread277)
     (at_kitchen_bread bread22)
     (at_kitchen_content content410)
     (at_kitchen_content content137)
     (no_gluten_bread bread277)
     (no_gluten_content content137)
     (allergic_gluten child239)
     (not_allergic_gluten child395)
     (waiting child395 table329)
     (waiting child239 table329)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child395)
     (served child239)
    )
  )
)
